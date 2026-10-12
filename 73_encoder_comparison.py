#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
73_encoder_comparison.py
========================
把「第二编码器对照」的结论固化：跨来源崩塌是否为**某一编码器特有问题**？

读取多个编码器的跨来源矩阵结果（`runs/_cross_source_matrix[_<encoder>]/results.json`），
汇总为对照表与图，并给出判定。

用法：
    python scripts/73_encoder_comparison.py
    python scripts/73_encoder_comparison.py --encoders phikon-v2 phikon vit-base-in
输出：
    runs/_encoder_comparison/results.json
    runs/_encoder_comparison/report.md
    docs/paper_figures/Figure5_encoder_comparison.{png,pdf}
    docs/paper_figures/Table7_encoder_comparison.{csv,md}
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

RUNS = ROOT / "runs"
FIGS = ROOT / "docs" / "paper_figures"
OUT = RUNS / "_encoder_comparison"

# 编码器元信息（维度与容量差异对判读很重要，必须一并报告）
ENCODER_META = {
    "phikon-v2": {"dim": 1024, "arch": "ViT-L/14", "params_m": 303.4,
                  "pretrain": "DINOv2 自监督，160k+ WSI", "open": True},
    "phikon": {"dim": 768, "arch": "ViT-B/16", "params_m": 86.4,
               "pretrain": "iBOT 自监督，40M TCGA 图像块", "open": True},
    "vit-base-in": {"dim": 768, "arch": "ViT-B/16", "params_m": 86.4,
                    "pretrain": "ImageNet 监督（非病理）", "open": True},
}

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 9, "figure.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})


def load_results(encoder: str) -> dict | None:
    d = RUNS / ("_cross_source_matrix" if encoder == "phikon-v2"
                else f"_cross_source_matrix_{encoder}")
    p = d / "results.json"
    if not p.exists():
        print(f"  [MISS] {encoder}: {p.relative_to(ROOT)}")
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _load_json(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def load_biopsy(encoder: str) -> dict | None:
    """外部活检验证：runs/_external_biopsy[_<encoder>]/results.json"""
    d = RUNS / ("_external_biopsy" if encoder == "phikon-v2"
                else f"_external_biopsy_{encoder}")
    return _load_json(d / "results.json")


def load_stain(encoder: str) -> dict | None:
    """染色归一化消融（同子集配对）：runs/_stain_ablation_encoders/results.json"""
    r = _load_json(RUNS / "_stain_ablation_encoders" / "results.json")
    if not r:
        return None
    return (r.get("summary") or {}).get(encoder)


def load_cross_specimen(encoder: str) -> dict | None:
    """跨标本类型：runs/_cross_specimen[_<encoder>]/"""
    d = RUNS / ("_cross_specimen" if encoder == "phikon-v2"
                else f"_cross_specimen_{encoder}")
    res = _load_json(d / "results.json")
    gap = _load_json(d / "calibration_gap.json")
    if res is None:
        return None
    out = {"results": res}
    if gap:
        strat = (gap.get("stratified_by_label") or {}).get("malignant") or {}
        out["stratified_malignant"] = {
            "mean_pred_in": strat.get("mean_pred_in"),
            "mean_pred_cross": strat.get("mean_pred_cross"),
            "delta_mean_pred": strat.get("delta_mean_pred"),
            "permutation_p": strat.get("permutation_p"),
            "ci95_delta": strat.get("ci95_delta"),
            "spread_in": strat.get("spread_in"),
            "spread_cross": strat.get("spread_cross"),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="*", default=list(ENCODER_META))
    args = ap.parse_args()

    data = {}
    for e in args.encoders:
        r = load_results(e)
        if r:
            data[e] = r
    if len(data) < 2:
        print(f"[ERROR] 至少需要 2 个编码器的结果（当前 {len(data)}）", file=sys.stderr)
        return 1

    targets = sorted(set().union(*[set(r["as_target"]) for r in data.values()]))
    summary: dict = {"encoders": {}, "gaps": {}, "n_cross": {}}

    for e, r in data.items():
        ind = r["in_domain"]
        cross = r["cross_matrix"]
        aucs = [v["auc"] for v in cross.values()]
        summary["encoders"][e] = {
            **ENCODER_META.get(e, {}),
            "n_sources": len(ind),
            "n_cross_pairs": len(cross),
            "in_domain_mean_auc": round(float(np.mean([v["auc"] for v in ind.values()])), 4),
            "in_domain_min_auc": round(float(np.min([v["auc"] for v in ind.values()])), 4),
            "cross_mean_auc": round(float(np.mean(aucs)), 4),
            "cross_min_auc": round(float(np.min(aucs)), 4),
            "cross_max_auc": round(float(np.max(aucs)), 4),
            "n_below_chance": int(sum(1 for a in aucs if a < 0.5)),
            "mean_gap": round(float(np.mean([r["as_target"][t]["gap"] for t in r["as_target"]])), 4),
        }
        summary["gaps"][e] = {t: r["as_target"][t]["gap"] for t in r["as_target"]}
        summary["n_cross"][e] = len(cross)

    # 判定：所有编码器是否都出现大幅崩塌
    gaps_all = [summary["encoders"][e]["mean_gap"] for e in data]
    collapsed = [g < -0.2 for g in gaps_all]
    summary["verdict"] = {
        "all_encoders_collapse": bool(all(collapsed)),
        "mean_gaps": {e: summary["encoders"][e]["mean_gap"] for e in data},
        "gap_range": [round(min(gaps_all), 4), round(max(gaps_all), 4)],
        "conclusion": (
            "跨来源崩塌在**所有被测编码器上一致出现**，因此不是某一模型的特有问题"
            if all(collapsed) else
            "崩塌幅度在不同编码器间不一致，需逐一定性讨论"
        ),
    }

    # ---- 其他三个实验的编码器维度汇总（这是「崩塌表现是否编码器依赖」的关键证据）
    extra: dict = {}
    for e in data:
        b = load_biopsy(e)
        s = load_stain(e)
        x = load_cross_specimen(e)
        rec: dict = {}
        if b:
            agg = b.get("aggregate", {})
            rec["biopsy"] = {
                "n_target": b.get("target_domain", {}).get("n_slides"),
                "sensitivity_at_0.5": (agg.get("sensitivity_at_0.5") or {}).get("mean"),
                "sensitivity_seed_sd": (agg.get("sensitivity_at_0.5") or {}).get("std"),
                "sensitivity_min": (agg.get("sensitivity_at_0.5") or {}).get("min"),
                "sensitivity_max": (agg.get("sensitivity_at_0.5") or {}).get("max"),
                "biopsy_mean_pred": (agg.get("biopsy_mean_pred") or {}).get("mean"),
                "biopsy_std_pred": (agg.get("biopsy_std_pred") or {}).get("mean"),
                "source_malignant_mean_pred": (agg.get("source_malignant_mean_pred") or {}).get("mean"),
                "source_val_auc": (agg.get("source_val_auc") or {}).get("mean"),
            }
        if s:
            rec["stain"] = {
                "cross_mean_original": s.get("cross_mean_original"),
                "cross_mean_stainnorm": s.get("cross_mean_stainnorm"),
                "delta_mean": s.get("delta_mean"),
                "n_improved_gt0.05": s.get("n_improved_gt0.05"),
                "n_worsened_gt0.05": s.get("n_worsened_gt0.05"),
            }
        if x:
            r = x["results"]
            sm = x.get("stratified_malignant") or {}
            rec["cross_specimen"] = {
                "in_domain_auc": (r.get("in_domain_tissue") or {}).get("auc"),
                "in_domain_ece": (r.get("in_domain_tissue") or {}).get("ece"),
                "cross_ece": (r.get("cross_specimen_diag") or {}).get("ece"),
                "cross_mean_pred": (r.get("cross_specimen_diag") or {}).get("mean_pred"),
                "stratified_delta_mean_pred": sm.get("delta_mean_pred"),
                "stratified_permutation_p": sm.get("permutation_p"),
                "stratified_ci95_delta": sm.get("ci95_delta"),
            }
        if rec:
            extra[e] = rec
    summary["other_experiments"] = extra

    # 活检域的「敏感度崩溃」是否编码器依赖 —— 这是本对照最有价值的发现
    sens = {e: extra[e]["biopsy"]["sensitivity_at_0.5"] for e in extra if "biopsy" in extra[e]}
    if len(sens) >= 2:
        vals = list(sens.values())
        spread = max(vals) - min(vals)
        # 均值之外还要看**种子间波动**：vit-base-in 的均值不低但波动远大于 phikon-v2，
        # 只看均值会把"不稳定"误读成"接近可用"。这里把两者一起报出来。
        seed_sd = {e: b.get("sensitivity_seed_sd") for e, b in
                   ((x, extra[x]["biopsy"]) for x in sens)
                   if b.get("sensitivity_seed_sd") is not None}
        summary["verdict"]["biopsy_sensitivity_encoder_dependent"] = bool(spread > 0.3)
        summary["verdict"]["biopsy_sensitivity_by_encoder"] = sens
        summary["verdict"]["biopsy_sensitivity_spread"] = round(spread, 4)
        summary["verdict"]["biopsy_sensitivity_seed_sd"] = seed_sd
        extra_note = ""
        if seed_sd and max(seed_sd.values()) > 0.05:
            worst = max(seed_sd, key=lambda k: seed_sd[k])
            extra_note = (f" 此外 {worst} 的活检敏感度在 3 个随机种子间的标准差达 "
                          f"{seed_sd[worst]:.4f}，远高于 "
                          f"{min(seed_sd, key=lambda k: seed_sd[k])} 的 "
                          f"{min(seed_sd.values()):.4f} —— 说明失效不是二元的："
                          "介于两者之间的编码器表现为**高方差**而非稳定可用。")
        summary["verdict"]["biopsy_note"] = (
            "活检域的决策阈值表现**强烈依赖编码器**（敏感度跨编码器相差 "
            f"{spread:.4f}），而 patch 级跨来源崩塌在各编码器上一致；"
            "说明「域内高 AUC」不足以预测下游可用性，且失效形式（崩塌 vs 阈值漂移）"
            "与编码器有关。" + extra_note
            if max(vals) - min(vals) > 0.3 else
            "活检域敏感度在各编码器间接近，未见明显的编码器依赖。")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- 图：各编码器「域内 vs 跨来源」，以及各目标域的落差对照
    encs = list(data)
    have_biopsy = [e for e in encs if "biopsy" in extra.get(e, {})]
    fig, axes = plt.subplots(1, 3 if have_biopsy else 2, figsize=(14.2 if have_biopsy else 10.6, 3.9),
                             gridspec_kw={"width_ratios": [1.0, 1.25, 1.15] if have_biopsy else [1.0, 1.25],
                                          "wspace": 0.40})

    ax = axes[0]
    x = np.arange(len(encs)); w = 0.36
    indv = [summary["encoders"][e]["in_domain_mean_auc"] for e in encs]
    crv = [summary["encoders"][e]["cross_mean_auc"] for e in encs]
    ax.bar(x - w / 2, indv, w, label="In-domain (mean)", color="#4c72b0")
    ax.bar(x + w / 2, crv, w, label="Cross-source (mean)", color="#c0392b")
    ax.axhline(0.5, color="black", ls=":", lw=0.9)
    ax.set_xticks(x); ax.set_xticklabels(encs, rotation=20, ha="right")
    ax.set_ylim(0, 1.08); ax.set_ylabel("AUC")
    ax.set_title("(a) All encoders: high in-domain, low cross-source")
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    for xi, (a, b) in enumerate(zip(indv, crv)):
        ax.text(xi - w / 2, a + 0.02, f"{a:.2f}", ha="center", fontsize=6.5)
        ax.text(xi + w / 2, b + 0.02, f"{b:.2f}", ha="center", fontsize=6.5)

    ax2 = axes[1]
    tgt = sorted(set().union(*[set(summary["gaps"][e]) for e in encs]))
    xx = np.arange(len(tgt)); w2 = 0.8 / max(len(encs), 1)
    colors = ["#4c72b0", "#dd8452", "#55a868", "#8172b3"]
    for i, e in enumerate(encs):
        vals = [summary["gaps"][e].get(t, np.nan) for t in tgt]
        ax2.bar(xx + (i - (len(encs) - 1) / 2) * w2, vals, w2, label=e, color=colors[i % len(colors)])
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_xticks(xx); ax2.set_xticklabels(tgt, rotation=20, ha="right")
    ax2.set_ylabel("ΔAUC (cross-source mean − in-domain)")
    ax2.set_title("(b) Performance drop by target source, per encoder")
    ax2.legend(frameon=False, fontsize=7)

    # (c) 活检域：源域恶性平均预测 vs 活检域平均预测 —— 阈值漂移是编码器依赖的
    if have_biopsy:
        ax3 = axes[2]
        yv = np.arange(len(have_biopsy))
        srcp = [extra[e]["biopsy"]["source_malignant_mean_pred"] for e in have_biopsy]
        tgp = [extra[e]["biopsy"]["biopsy_mean_pred"] for e in have_biopsy]
        ax3.barh(yv + 0.19, srcp, 0.36, label="Source domain (TCGA malignant)", color="#4c72b0")
        ax3.barh(yv - 0.19, tgp, 0.36, label="Biopsy domain (ENDO-AID)", color="#c0392b")
        ax3.axvline(0.5, color="black", ls="--", lw=1.1)
        for i, (a, b) in enumerate(zip(srcp, tgp)):
            ax3.text(a + 0.015, i + 0.19, f"{a:.3f}", va="center", fontsize=6.5)
            ax3.text(b + 0.015, i - 0.19, f"{b:.3f}", va="center", fontsize=6.5)
        ax3.set_yticks(yv)
        ax3.set_yticklabels([f"{e}\nsens@0.5 = {extra[e]['biopsy']['sensitivity_at_0.5']:.3f}"
                             for e in have_biopsy], fontsize=7)
        ax3.set_xlim(0, 1.30); ax3.set_xlabel("Mean predicted probability of malignancy")
        ax3.set_title("(c) Biopsy failure mode is encoder-dependent")
        # 图例放在上方空白处：放在下方会与 phikon 的红色短条重叠
        ax3.legend(frameon=False, fontsize=6.5, loc="upper center", ncol=2,
                   bbox_to_anchor=(0.5, 1.02), handlelength=1.2, columnspacing=1.0)

    fig.suptitle("Cross-source collapse is consistent across independently trained encoders; "
                 "the biopsy decision-threshold failure is not",
                 fontsize=10.5, y=1.03)
    FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"Figure5_encoder_comparison.{ext}", dpi=300)
    plt.close(fig)

    # ---- 表
    header = ["encoder", "arch", "dim", "params_M", "pretraining",
              "in_domain_mean_AUC", "cross_mean_AUC", "cross_range", "mean_delta_AUC", "pairs_below_chance"]
    rows = []
    for e in encs:
        s = summary["encoders"][e]
        rows.append([e, s.get("arch", ""), s.get("dim", ""), s.get("params_m", ""),
                     s.get("pretrain", ""), s["in_domain_mean_auc"], s["cross_mean_auc"],
                     f"{s['cross_min_auc']}–{s['cross_max_auc']}", s["mean_gap"],
                     f"{s['n_below_chance']}/{s['n_cross_pairs']}"])
    with open(FIGS / "Table7_encoder_comparison.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh); w.writerow(header); w.writerows(rows)
    md = ["# Table 7. Cross-source transfer across independently trained encoders", "",
          "| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for r in rows:
        md.append("| " + " | ".join(str(c) for c in r) + " |")
    md += ["", "> All encoders are openly available (no gated access required). "
              "Capacities differ (ViT-L vs ViT-B), which is stated explicitly: the point is the "
              "*consistency of the collapse*, not an encoder ranking.",
           "> Phikon-v2↔phikon share the same publisher and pretraining family; "
           "vit-base-in is an ImageNet-supervised, non-pathology control."]
    (FIGS / "Table7_encoder_comparison.md").write_text("\n".join(md), encoding="utf-8")

    # ---- Table 8：跨实验的编码器对照（崩塌一致 vs 阈值行为依赖编码器）
    h8 = ["encoder", "patch_in_domain_mean_AUC", "patch_cross_mean_AUC", "patch_mean_delta_AUC",
          "stain_delta_mean_AUC", "biopsy_sensitivity@0.5", "biopsy_sens_seed_SD",
          "biopsy_mean_prob", "source_malignant_mean_prob",
          "cross_specimen_delta_prob", "cross_specimen_perm_p"]
    r8 = []
    for e in encs:
        rec = extra.get(e, {})
        b, s, x = rec.get("biopsy"), rec.get("stain"), rec.get("cross_specimen")
        r8.append([
            e,
            summary["encoders"][e]["in_domain_mean_auc"],
            summary["encoders"][e]["cross_mean_auc"],
            summary["encoders"][e]["mean_gap"],
            s["delta_mean"] if s else "n/a",
            b["sensitivity_at_0.5"] if b else "n/a",
            b.get("sensitivity_seed_sd") if b else "n/a",
            b["biopsy_mean_pred"] if b else "n/a",
            b["source_malignant_mean_pred"] if b else "n/a",
            x["stratified_delta_mean_pred"] if x else "n/a",
            x["stratified_permutation_p"] if x else "n/a",
        ])
    with open(FIGS / "Table8_encoder_by_experiment.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh); w.writerow(h8); w.writerows(r8)
    md8 = ["# Table 8. Encoder dependence across all four experiments", "",
           "| " + " | ".join(h8) + " |", "| " + " | ".join("---" for _ in h8) + " |"]
    for r in r8:
        md8.append("| " + " | ".join(str(c) for c in r) + " |")
    md8 += ["", "> `patch_*` = 5 patch sources, logistic probe on frozen features; "
                "`stain_delta_mean_AUC` = mean paired ΔAUC after Reinhard normalisation on an identical image subset; "
                "`biopsy_*` = 91 ENDO-AID Pipelle biopsies, source domain = TCGA frozen tissue sections "
                "(97 slides for all three encoders; only `tcga_normal` + `tcga_tumor` features are required); "
                "`cross_specimen_*` = label-stratified (malignant only) shift in mean predicted probability between "
                "tissue-section and FFPE diagnostic slides.",
            "> Reading: the *collapse* is encoder-independent (patch), while the *decision-threshold failure* "
            "in the biopsy domain is strongly encoder-dependent, and is **not binary** — the intermediate encoder "
            "(`vit-base-in`) reaches a lower sensitivity with a seed-to-seed SD an order of magnitude larger "
            "than `phikon-v2`'s. `n/a` = experiment not run for that encoder."]
    (FIGS / "Table8_encoder_by_experiment.md").write_text("\n".join(md8), encoding="utf-8")

    # ---- 报告
    lines = ["# 第二编码器对照：跨来源崩塌是否为编码器特有问题", "",
             f"被测编码器 {len(encs)} 个（**全部为开放权重，无需任何申请**）", "",
             "## 1. 各编码器汇总", "",
             "| 编码器 | 架构 | 维度 | 参数量 | 预训练 | 域内平均 AUC | 跨来源平均 AUC | 平均落差 | 低于随机 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for e in encs:
        s = summary["encoders"][e]
        lines.append(f"| {e} | {s.get('arch','')} | {s.get('dim','')} | {s.get('params_m','')}M | "
                     f"{s.get('pretrain','')} | {s['in_domain_mean_auc']:.4f} | "
                     f"**{s['cross_mean_auc']:.4f}** | **{s['mean_gap']:+.4f}** | "
                     f"{s['n_below_chance']}/{s['n_cross_pairs']} |")
    v = summary["verdict"]
    lines += ["", "## 2. 判定", "", f"**{v['conclusion']}**", "",
              f"- 各编码器平均落差范围：**{v['gap_range'][0]:+.4f} 至 {v['gap_range'][1]:+.4f}**",
              f"- 是否全部出现大幅崩塌（落差 < −0.2）：**{v['all_encoders_collapse']}**",
              "", "逐目标域落差：", "",
              "| 目标域 | " + " | ".join(encs) + " |",
              "| --- | " + " | ".join("---" for _ in encs) + " |"]
    for t in tgt:
        lines.append(f"| {t} | " + " | ".join(
            f"{summary['gaps'][e].get(t, float('nan')):+.4f}" for e in encs) + " |")
    lines += ["", "## 3. 跨实验的编码器维度", "",
              "| 编码器 | patch 跨来源 ΔAUC | 染色归一化 ΔAUC 均值 | 活检敏感度@0.5 | 活检平均预测 | 源域恶性平均预测 | 跨标本类型 Δ概率 | 置换 p |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for e in encs:
        rec = extra.get(e, {})
        b, s, x = rec.get("biopsy"), rec.get("stain"), rec.get("cross_specimen")
        def _f(v, fmt="{:.4f}"):
            return fmt.format(v) if isinstance(v, (int, float)) else "n/a"
        lines.append(
            f"| {e} | {summary['encoders'][e]['mean_gap']:+.4f} | {_f(s['delta_mean'] if s else None, '{:+.4f}')} | "
            f"**{_f(b['sensitivity_at_0.5'] if b else None)}** | {_f(b['biopsy_mean_pred'] if b else None)} | "
            f"{_f(b['source_malignant_mean_pred'] if b else None)} | "
            f"{_f(x['stratified_delta_mean_pred'] if x else None, '{:+.4f}')} | "
            f"{_f(x['stratified_permutation_p'] if x else None)} |")
    if "biopsy_note" in summary["verdict"]:
        lines += ["", f"**关键判读**：{summary['verdict']['biopsy_note']}"]

    lines += ["", "## 4. 必须写明的局限", "",
              "- 三个编码器**容量不同**（Phikon-v2 是 ViT-L，phikon 与 vit-base-in 是 ViT-B），"
              "因此本对照检验的是「**崩塌现象是否普遍**」，**不是**编码器性能排名；",
              "- 未纳入 gated 模型（UNI2-h / CONCH / Virchow2 / H-optimus / Hibou），"
              "它们需机构邮箱授权；**本结论的适用范围限于可公开获取的编码器**；",
              "- phikon 与 phikon-v2 同属 Owkin 系，相关性较高；vit-base-in 是非病理对照，"
              "三者共同提供了「病理自监督 + 非病理监督」两类证据。"]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(summary["verdict"], indent=2, ensure_ascii=False))
    for e in encs:
        s = summary["encoders"][e]
        print(f"  {e:14s} 域内 {s['in_domain_mean_auc']:.4f}  跨来源 {s['cross_mean_auc']:.4f}  "
              f"落差 {s['mean_gap']:+.4f}  低于随机 {s['n_below_chance']}/{s['n_cross_pairs']}")
    print(f"\n-> {OUT/'report.md'}")
    print(f"-> {FIGS/'Figure5_encoder_comparison.png'}")
    print(f"-> {FIGS/'Table7_encoder_comparison.md'}")
    print(f"-> {FIGS/'Table8_encoder_by_experiment.md'}")
    if "biopsy_note" in summary["verdict"]:
        print(f"\n活检验证（按编码器）: {summary['verdict']['biopsy_sensitivity_by_encoder']}")
        print(f"  {summary['verdict']['biopsy_note']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
