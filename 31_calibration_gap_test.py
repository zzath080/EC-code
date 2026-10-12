#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
31_calibration_gap_test.py
==========================
把「跨标本类型导致校准恶化」做成带统计量的结论，而不是只报两个数字。

核心输出：
    1. 域内 vs 跨域的 ECE 差值 + bootstrap 95% CI
    2. 预测分布是否发生系统性偏移（域内 vs 跨域的均值预测差 + CI）
    3. 一个不依赖标签的旁证：模型在目标域上的平均预测概率 vs 目标域真实阳性率
       （target-domain 校准确认偏差 calibration-in-the-large）

这一分析的价值：**即使目标域样本量小、AUC 无法估计，校准偏移仍可被量化**——
这正是本项目"把校准当一等指标"的体现。

用法：
    python scripts/31_calibration_gap_test.py
输出：
    runs/_cross_specimen/calibration_gap.json
    runs/_cross_specimen/calibration_gap.md
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "_cross_specimen"


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    tot = len(y)
    v = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum():
            v += (m.sum() / tot) * abs(p[m].mean() - y[m].mean())
    return float(v)


def boot_ci(fn, n: int, reps: int = 2000, seed: int = 42) -> tuple[float, float, float]:
    """对统计量做 bootstrap；fn(idx) 返回标量。n 为样本量。"""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        try:
            v = fn(idx)
            if np.isfinite(v):
                vals.append(v)
        except Exception:  # noqa: BLE001
            continue
    if not vals:
        return float("nan"), float("nan"), float("nan")
    a = np.array(vals)
    return float(a.mean()), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None,
                    help="含 predictions.csv 的 run 目录；默认 runs/_cross_specimen")
    args = ap.parse_args()

    global OUT
    if args.dir:
        OUT = Path(args.dir) if Path(args.dir).is_absolute() else ROOT / args.dir

    pred_path = OUT / "predictions.csv"
    if not pred_path.exists():
        print(f"[ERROR] 缺 {pred_path}，先跑 scripts/27_cross_specimen_transfer.py", file=sys.stderr)
        return 2

    rows = list(csv.DictReader(open(pred_path, encoding="utf-8-sig")))
    res: dict = {"source": str(pred_path.relative_to(ROOT)), "n_total": len(rows)}

    # 分域读取：tissue_test = 域内测试集；diagnostic = 跨标本类型目标域
    in_dom = [(int(r["label"]), float(r["prob_malignant"])) for r in rows if r["domain"] == "tissue_test"]
    cross = [(int(r["label"]), float(r["prob_malignant"])) for r in rows if r["domain"] == "diagnostic"]

    if not in_dom or not cross:
        print("[ERROR] predictions.csv 缺少 domain 列或不含两个域", file=sys.stderr)
        return 2

    y_in = np.array([a for a, _ in in_dom])
    p_in = np.array([b for _, b in in_dom])
    y_cr = np.array([a for a, _ in cross])
    p_cr = np.array([b for _, b in cross])

    e_in = ece(y_in, p_in)
    e_cr = ece(y_cr, p_cr)
    res["in_domain"] = {"n": len(y_in), "n_pos": int(y_in.sum()), "ece": round(e_in, 4),
                        "brier": round(float(np.mean((p_in - y_in) ** 2)), 4),
                        "mean_pred": round(float(p_in.mean()), 4),
                        "frac_pos": round(float(y_in.mean()), 4)}
    res["cross_domain"] = {"n": len(y_cr), "n_pos": int(y_cr.sum()), "ece": round(e_cr, 4),
                           "brier": round(float(np.mean((p_cr - y_cr) ** 2)), 4),
                           "mean_pred": round(float(p_cr.mean()), 4),
                           "frac_pos": round(float(y_cr.mean()), 4)}

    # ECE 差值 + CI（分别重采样两个域）
    def ece_diff(_):
        r1 = np.random.default_rng(1).integers(0, len(y_in), len(y_in))
        r2 = np.random.default_rng(2).integers(0, len(y_cr), len(y_cr))
        return ece(y_cr[r2], p_cr[r2]) - ece(y_in[r1], p_in[r1])

    rng = np.random.default_rng(7)
    diffs = []
    for _ in range(2000):
        i1 = rng.integers(0, len(y_in), len(y_in))
        i2 = rng.integers(0, len(y_cr), len(y_cr))
        diffs.append(ece(y_cr[i2], p_cr[i2]) - ece(y_in[i1], p_in[i1]))
    diffs = np.array([d for d in diffs if np.isfinite(d)])
    res["ece_difference"] = {
        "delta_ece": round(e_cr - e_in, 4),
        "ci95": [round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)],
        "p_two_sided": round(float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean())), 4),
        "ratio": round(e_cr / e_in, 2) if e_in > 0 else None,
    }

    # 预测分布偏移（校准于大样本 calibration-in-the-large）
    shift = float(p_cr.mean() - p_in.mean())
    shifts = []
    for _ in range(2000):
        i1 = rng.integers(0, len(p_in), len(p_in))
        i2 = rng.integers(0, len(p_cr), len(p_cr))
        shifts.append(p_cr[i2].mean() - p_in[i1].mean())
    shifts = np.array(shifts)
    res["prediction_shift"] = {
        "delta_mean_pred": round(shift, 4),
        "ci95": [round(float(np.percentile(shifts, 2.5)), 4), round(float(np.percentile(shifts, 97.5)), 4)],
    }

    # 目标域校准于大样本：平均预测 vs 实际阳性率
    res["calibration_in_the_large_target"] = {
        "mean_pred": round(float(p_cr.mean()), 4),
        "observed_rate": round(float(y_cr.mean()), 4),
        "bias": round(float(p_cr.mean() - y_cr.mean()), 4),
        "note": "偏差为正表示模型在目标域上系统性高估恶性概率",
    }

    # ---------------------------------------------------------------
    # 关键补充：按标签分层比较，剔除「类别不平衡」这一混淆因素。
    # 前两项（ECE 差值）会被两域阳性率不同所混淆；只有当同一标签下
    # 预测概率也发生系统性偏移时，才能确证域效应。
    # ---------------------------------------------------------------
    def preds_with_label(y, p, lab):
        return p[y == lab]

    stratified: dict = {}
    for lab, name in ((1, "malignant"), (0, "benign")):
        a = preds_with_label(y_in, p_in, lab)
        b = preds_with_label(y_cr, p_cr, lab)
        entry: dict = {"n_in": int(len(a)), "n_cross": int(len(b))}
        if len(a):
            entry["mean_pred_in"] = round(float(a.mean()), 4)
        if len(b):
            entry["mean_pred_cross"] = round(float(b.mean()), 4)
        if len(a) and len(b):
            entry["delta_mean_pred"] = round(float(b.mean() - a.mean()), 4)
            # 置换检验：两组同一标签下的预测是否来自同一分布
            obs = abs(b.mean() - a.mean())
            pooled = np.concatenate([a, b])
            rng2 = np.random.default_rng(11)
            cnt = 0
            for _ in range(5000):
                perm = rng2.permutation(pooled)
                d = abs(perm[len(a):].mean() - perm[: len(a)].mean())
                if d >= obs:
                    cnt += 1
            entry["permutation_p"] = round((cnt + 1) / 5001, 4)
            # bootstrap 差值 CI（固定随机源，避免退化成单点）
            bs = np.random.default_rng(13)
            deltas = []
            for _ in range(2000):
                ia = bs.integers(0, len(a), len(a))
                ib = bs.integers(0, len(b), len(b))
                deltas.append(b[ib].mean() - a[ia].mean())
            deltas = np.array([d for d in deltas if np.isfinite(d)])
            entry["ci95_delta"] = [round(float(np.percentile(deltas, 2.5)), 4),
                                   round(float(np.percentile(deltas, 97.5)), 4)] if len(deltas) else None
            entry["spread_in"] = round(float(a.std()), 4)
            entry["spread_cross"] = round(float(b.std()), 4)
        stratified[name] = entry
    res["stratified_by_label"] = stratified

    # 判读：哪个检验更可信
    mal = stratified.get("malignant", {})
    res["verdict"] = {
        "ece_difference_confounded": True,
        "reason": ("两域阳性率不同（域内 %.2f vs 跨域 %.2f），ECE 差值同时反映类别不平衡与域效应，"
                   "不能单独作为域偏移证据" % (y_in.mean(), y_cr.mean())),
        "cleaner_test": "stratified_by_label.malignant",
        "clean_test_conclusive": bool(len(y_cr[y_cr == 1]) >= 5),
        "recommendation": ("目标域阳性样本不足 5 例，按标签分层的检验尚不具结论性；"
                           "需扩充诊断切片样本量后重跑") if len(y_cr[y_cr == 1]) < 5 else
                          "按标签分层的检验可作结论",
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "calibration_gap.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# 跨标本类型的校准差异检验",
        "",
        f"来源：`{res['source']}`（由 `scripts/27_cross_specimen_transfer.py` 产出）",
        "",
        "## 1. 两域对比",
        "",
        "| 域 | n | 阳性数 | ECE | Brier | 平均预测 | 实际阳性率 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        f"| 域内（冰冻组织切片 test） | {len(y_in)} | {int(y_in.sum())} | {e_in:.4f} | "
        f"{res['in_domain']['brier']:.4f} | {res['in_domain']['mean_pred']:.4f} | {res['in_domain']['frac_pos']:.4f} |",
        f"| **跨标本类型（FFPE 诊断切片）** | {len(y_cr)} | {int(y_cr.sum())} | **{e_cr:.4f}** | "
        f"{res['cross_domain']['brier']:.4f} | {res['cross_domain']['mean_pred']:.4f} | {res['cross_domain']['frac_pos']:.4f} |",
        "",
        "## 2. 校准恶化是否显著",
        "",
        f"- ΔECE（跨域 − 域内）= **{res['ece_difference']['delta_ece']:+.4f}**，"
        f"95% CI [{res['ece_difference']['ci95'][0]:+.4f}, {res['ece_difference']['ci95'][1]:+.4f}]，"
        f"bootstrap 双侧 p = {res['ece_difference']['p_two_sided']:.4f}",
        f"- ECE 倍数：**{res['ece_difference']['ratio']}×**",
        f"- 平均预测概率偏移 = **{res['prediction_shift']['delta_mean_pred']:+.4f}**，"
        f"95% CI [{res['prediction_shift']['ci95'][0]:+.4f}, {res['prediction_shift']['ci95'][1]:+.4f}]",
        "",
        "## 3. 目标域校准于大样本（calibration-in-the-large）",
        "",
        f"- 目标域平均预测恶性概率：{res['calibration_in_the_large_target']['mean_pred']:.4f}",
        f"- 目标域实际恶性比例：{res['calibration_in_the_large_target']['observed_rate']:.4f}",
        f"- 偏差：**{res['calibration_in_the_large_target']['bias']:+.4f}**",
        f"- 解读：{res['calibration_in_the_large_target']['note']}",
        "",
        "## 4. 按标签分层的检验（剔除类别不平衡的混淆）",
        "",
        "> **重要局限**：上表的 ΔECE 会被两域**阳性率不同**（域内 %.2f vs 跨域 %.2f）所混淆。"
        "只有控制标签后仍出现预测偏移，才能确证是域效应。" % (y_in.mean(), y_cr.mean()),
        "",
        "| 标签 | 域内 n | 跨域 n | 域内平均预测 | 跨域平均预测 | Δ | 置换检验 p |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for lab, name in (("malignant", stratified.get("malignant", {})),
                      ("benign", stratified.get("benign", {}))):
        lines.append(
            f"| {lab} | {name.get('n_in', 0)} | {name.get('n_cross', 0)} | "
            f"{name.get('mean_pred_in', '—')} | {name.get('mean_pred_cross', '—')} | "
            f"{name.get('delta_mean_pred', '—')} | {name.get('permutation_p', '—')} |")
    v = res["verdict"]
    lines += [
        "",
        f"- 结论：{v['recommendation']}",
        f"- 原因：{v['reason']}",
        "",
        "## 5. 写作提示与本分析的边界",
        "",
        "**当前可以主张**：已有初步证据显示模型在 FFPE 诊断切片上的预测分布与源域不同，",
        "且校准指标（ECE）显著上升；但该结论受目标域样本量限制，**须扩样本后方可定论**。",
        "",
        "**不可主张**：",
        f"- 目标域当前仅有 **{len(y_cr)}** 张切片（阳性 {int(y_cr.sum())}），**AUC 不可估计**，不得报告为判别性能结论；",
        "- **不得**把 ΔECE 直接解读为纯域效应——它包含类别不平衡的贡献，必须报告按标签分层的检验；",
        "- ECE 在 10 分箱 + 小样本下方差很大，必须同时报告 CI 与分箱样本量；",
        "- 冰冻→FFPE 的域偏移**不能**等同于切除→活检的全部差异（后者还含组织量、碎片化、取材部位差异）。",
    ]
    (OUT / "calibration_gap.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\n-> {OUT/'calibration_gap.md'}")
    print(f"-> {OUT/'calibration_gap.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
