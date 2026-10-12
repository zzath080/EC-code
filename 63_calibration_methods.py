#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
63_calibration_methods.py
=========================
校准方法对比与**跨来源校准迁移检验**。

要回答的问题：
    Q1 在域内，温度缩放 / 分位数（isotonic）校准能把 ECE 降到多少？
    Q2 **在源域拟合的校准参数，迁到目标域还有效吗？**
        —— 这是本文的关切点：若校准也不可迁移，则"跨来源"问题比判别性能更深。

做法（对每一对 源→目标）：
    1. 用源域训练线性分类器；
    2. 在源域的**留出集**上拟合三种校准：无校准 / 温度缩放 / isotonic；
    3. 分别在源域留出集与目标域上评估 ECE、Brier、AUC；
    4. 对比"源域拟合校准后用在目标域" 与 "在目标域自身上拟合校准（上界参考）"。

判读：
    - 若源域拟合的校准在目标域上**无改善甚至恶化** → 校准不可迁移；
    - 若目标域自身拟合校准改善很大 → 说明只是参数不匹配，需要目标域标签。

用法：
    python scripts/63_calibration_methods.py
输出：
    runs/_calibration/results.json
    runs/_calibration/report.md
    runs/_calibration/calibration_table.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / ".pylibs"
if LIB.exists():
    sys.path.insert(0, str(LIB))

FEAT = ROOT / "data" / "features" / "phikon-v2" / "patch_multisource"
OUT = ROOT / "runs" / "_calibration"


def ece(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    tot = len(y)
    v = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum():
            v += (m.sum() / tot) * abs(float(p[m].mean()) - float(y[m].mean()))
    return float(v)


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def temperature_scale(logits: np.ndarray, y: np.ndarray) -> float:
    """在留出集上最小化 NLL 求温度 T（单参数，网格搜索以避免依赖 scipy 优化器）。"""
    best_t, best_nll = 1.0, float("inf")
    for t in np.linspace(0.05, 10.0, 200):
        z = logits / t
        p = 1 / (1 + np.exp(-z))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        nll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        if nll < best_nll:
            best_nll, best_t = nll, float(t)
    return best_t


def isotonic_fit(p: np.ndarray, y: np.ndarray):
    """分位数（isotonic）校准：返回可作用于新概率的映射函数。"""
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    ir.fit(p, y)
    return ir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", type=float, default=0.3, help="源域留出用于拟合校准的比例")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    data = {}
    for npz in sorted(FEAT.glob("*.npz")):
        d = np.load(npz, allow_pickle=True)
        data[npz.stem] = (d["features"], d["labels"].astype(int))
    names = sorted(data)
    print(f"来源 {len(names)} 个：" + ", ".join(names))
    if len(names) < 2:
        print("[ERROR] 至少需要 2 个来源", file=sys.stderr)
        return 1

    rows: list[dict] = []
    for src in names:
        Xs, ys = data[src]
        rng = np.random.default_rng(args.seed)
        idx = rng.permutation(len(ys))
        n_fit = int(len(ys) * (1 - args.holdout))
        i_fit, i_cal = idx[:n_fit], idx[n_fit:]

        # 源域分类器
        clf = LogisticRegression(max_iter=2000).fit(Xs[i_fit], ys[i_fit])
        # 用决策函数（logit）以便温度缩放
        z_cal = clf.decision_function(Xs[i_cal])
        p_cal = 1 / (1 + np.exp(-z_cal))
        y_cal = ys[i_cal]

        T = temperature_scale(z_cal, y_cal)
        p_cal_T = 1 / (1 + np.exp(-(z_cal / T)))
        ir = isotonic_fit(p_cal, y_cal)

        for tgt in names:
            if tgt == src:
                continue
            Xt, yt = data[tgt]
            z_t = clf.decision_function(Xt)
            p_t = 1 / (1 + np.exp(-z_t))

            row = {
                "source": src, "target": tgt,
                "n_target": len(yt), "target_pos_rate": round(float(yt.mean()), 4),
                "auc": round(float(roc_auc_score(yt, p_t)), 4),
                # 未校准
                "ece_raw": round(ece(yt, p_t), 4),
                "brier_raw": round(brier(yt, p_t), 4),
                "mean_pred_raw": round(float(p_t.mean()), 4),
                # 源域拟合的温度缩放迁到目标域
                "T_source": round(T, 3),
                "ece_T_from_source": round(ece(yt, 1 / (1 + np.exp(-(z_t / T)))), 4),
                "brier_T_from_source": round(brier(yt, 1 / (1 + np.exp(-(z_t / T)))), 4),
                # 源域拟合的 isotonic 迁到目标域
                "ece_iso_from_source": round(ece(yt, ir.predict(p_t)), 4),
                "brier_iso_from_source": round(brier(yt, ir.predict(p_t)), 4),
                # 上界参考：在目标域自身（用其全部标签）拟合 isotonic
                "ece_iso_oracle": round(ece(yt, isotonic_fit(p_t, yt).predict(p_t)), 4),
                # 目标域自身的最优温度（上界参考）
                "T_target_oracle": round(temperature_scale(clf.decision_function(Xt), yt), 3),
            }
            row["ece_T_target_oracle"] = round(
                ece(yt, 1 / (1 + np.exp(-(row["T_target_oracle"] and
                                          clf.decision_function(Xt) / row["T_target_oracle"])))), 4)
            rows.append(row)
            print(f"  {src:16s} -> {tgt:16s} AUC={row['auc']:.3f}  "
                  f"ECE raw={row['ece_raw']:.3f}  T(from src)={row['ece_T_from_source']:.3f}  "
                  f"iso(from src)={row['ece_iso_from_source']:.3f}  iso(目标域自拟合)={row['ece_iso_oracle']:.3f}")

    # 汇总
    def mean_of(key: str) -> float:
        vals = [r[key] for r in rows if np.isfinite(r[key])]
        return round(float(np.mean(vals)), 4) if vals else float("nan")

    summary = {
        "n_pairs": len(rows),
        "mean_ece_raw": mean_of("ece_raw"),
        "mean_ece_temperature_from_source": mean_of("ece_T_from_source"),
        "mean_ece_isotonic_from_source": mean_of("ece_iso_from_source"),
        "mean_ece_isotonic_oracle": mean_of("ece_iso_oracle"),
        "mean_brier_raw": mean_of("brier_raw"),
        "mean_brier_temperature_from_source": mean_of("brier_T_from_source"),
        "mean_brier_isotonic_from_source": mean_of("brier_iso_from_source"),
    }
    # 判读
    dT = summary["mean_ece_temperature_from_source"] - summary["mean_ece_raw"]
    dI = summary["mean_ece_isotonic_from_source"] - summary["mean_ece_raw"]
    dO = summary["mean_ece_isotonic_oracle"] - summary["mean_ece_raw"]
    if dT < -0.01 or dI < -0.01:
        verdict = "源域拟合的校准可部分迁移（跨来源 ECE 有改善）"
    else:
        verdict = ("源域拟合的校准**不能迁移**（跨来源 ECE 无改善或恶化）——"
                   "说明校准失效不是简单参数错配，需要目标域标签才能修正")
    summary["verdict"] = verdict
    summary["delta_ece_temperature"] = round(dT, 4)
    summary["delta_ece_isotonic"] = round(dI, 4)
    summary["delta_ece_isotonic_oracle"] = round(dO, 4)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(
        json.dumps({"summary": summary, "pairs": rows}, indent=2, ensure_ascii=False), encoding="utf-8")

    fields = list(rows[0].keys())
    with open(OUT / "calibration_table.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    lines = ["# 校准方法对比与跨来源校准迁移检验", "",
             f"来源对数：{len(rows)}（每对为「源域训练 → 目标域测试」）", "",
             "## 1. 逐对结果", "",
             "| 源→目标 | AUC | ECE(原始) | ECE(温度缩放，源域拟合) | ECE(isotonic，源域拟合) | "
             "ECE(isotonic，目标域自拟合=上界) |",
             "| --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        lines.append(f"| {r['source']}→{r['target']} | {r['auc']:.3f} | {r['ece_raw']:.4f} | "
                     f"{r['ece_T_from_source']:.4f} | {r['ece_iso_from_source']:.4f} | "
                     f"{r['ece_iso_oracle']:.4f} |")
    lines += ["", "## 2. 平均 ECE", "",
              "| 校准方式 | 平均 ECE | 相对原始的 ΔECE |", "| --- | --- | --- |",
              f"| 无校准 | {summary['mean_ece_raw']:.4f} | — |",
              f"| 温度缩放（源域拟合，迁到目标域） | {summary['mean_ece_temperature_from_source']:.4f} | "
              f"{summary['delta_ece_temperature']:+.4f} |",
              f"| isotonic（源域拟合，迁到目标域） | {summary['mean_ece_isotonic_from_source']:.4f} | "
              f"{summary['delta_ece_isotonic']:+.4f} |",
              f"| isotonic（目标域自拟合，上界参考） | {summary['mean_ece_isotonic_oracle']:.4f} | "
              f"{summary['delta_ece_isotonic_oracle']:+.4f} |",
              "", "## 3. 判读", "", f"{verdict}", "",
              f"- 温度缩放迁移后 ΔECE = **{summary['delta_ece_temperature']:+.4f}**",
              f"- isotonic 迁移后 ΔECE = **{summary['delta_ece_isotonic']:+.4f}**",
              f"- 目标域自拟合 isotonic（上界）ΔECE = **{summary['delta_ece_isotonic_oracle']:+.4f}**",
              "",
              "> 若「源域拟合」与「目标域自拟合」差距很大，说明：**校准问题的根源是域不匹配，",
              "> 而非校准算法本身**；临床部署时若没有目标域标签，就无法修正这一偏差。",
              "",
              "## 4. 局限", "",
              "> ⚠️ **必须注意的判读陷阱**：本表中原始 ECE 已高达 0.35 量级，而同一批配对的",
              "> 跨来源 AUC 仅 0.35–0.82（见 `runs/_cross_source_matrix/report.md`）。",
              "> 当 AUC 接近随机时，**概率与标签几乎无关**，此时 ECE 反映的主要是",
              "> 「判别力丧失」而非单纯的「校准不良」。因此：",
              "> - 不应把「校准不可迁移」单独解读为独立的科学发现，它与判别力崩塌是同一现象的",
              ">   两个侧面；",
              "> - 更稳妥的表述是：**跨来源时模型既失去了判别力，也失去了可迁移的置信度**。",
              "",
              "- 二分类校准只用了两个参数族（温度缩放、isotonic），未测试 Dirichlet 校准、",
              "  向量缩放等；",
              "- 目标域「自拟合」是在全部目标域数据上拟合再评估（in-sample），**不是**无偏上界，",
              "  仅作为「最好能做到多少」的粗略参考——实测其 ECE≈0.0001 正是 in-sample 过拟合的表现，",
              "  **不得**当作可达到的性能；",
              "- 各来源类别定义不同（见主设计文档 §3.1 局限）。"]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n" + json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n-> {OUT/'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
