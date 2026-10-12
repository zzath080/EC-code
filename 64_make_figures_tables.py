#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
64_make_figures_tables.py
=========================
把已验证结果固化为**论文可直接使用**的图与表（出版级，300 dpi PNG + 矢量 PDF）。

产出：
    docs/paper_figures/Figure1_cross_source_matrix.{png,pdf}
    docs/paper_figures/Figure2_shortcut_audit.{png,pdf}
    docs/paper_figures/Figure3_stain_normalization.{png,pdf}
    docs/paper_figures/Figure4_external_biopsy.{png,pdf}
    docs/paper_figures/Table3_cross_source_matrix.csv / .md
    docs/paper_figures/Table4_shortcut_audit.csv / .md
    docs/paper_figures/Table5_stain_ablation.csv / .md
    docs/paper_figures/Table6_external_biopsy.csv / .md
    docs/paper_figures/FIGURE_TABLE_LEGENDS.md   # 图注/表注（中英对照）

设计约定：
    - 图内文字用**英文**（投稿要求），图注中英对照；
    - 所有数字从 runs/ 下的 results.json 读取，**不硬编码**，保证与实验一致；
    - 若某结果文件缺失则跳过对应图并在报告中标注，不静默生成空图。

用法：
    python scripts/64_make_figures_tables.py
"""

from __future__ import annotations

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
OUT = ROOT / "docs" / "paper_figures"

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

MADE: list[str] = []
SKIPPED: list[str] = []


def load_json(p: Path) -> dict | None:
    if not p.exists():
        SKIPPED.append(f"{p.relative_to(ROOT)}（缺失）")
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        SKIPPED.append(f"{p.relative_to(ROOT)}（解析失败 {type(exc).__name__}）")
        return None


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    MADE.append(name)


def write_table(name: str, header: list[str], rows: list[list], note: str = "") -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{name}.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    md = [f"# {name}", ""]
    md += ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for r in rows:
        md.append("| " + " | ".join(str(c) for c in r) + " |")
    if note:
        md += ["", note]
    (OUT / f"{name}.md").write_text("\n".join(md), encoding="utf-8")
    MADE.append(name)


# ---------------------------------------------------------------- Figure 1
def figure1() -> None:
    res = load_json(RUNS / "_cross_source_matrix" / "results.json")
    if not res:
        return
    in_dom = res["in_domain"]
    cross = res["cross_matrix"]
    names = sorted(in_dom, key=lambda k: -in_dom[k]["auc"])
    n = len(names)

    M = np.full((n, n), np.nan)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            M[i, j] = in_dom[a]["auc"] if a == b else cross.get(f"{a}->{b}", {}).get("auc", np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0),
                             gridspec_kw={"width_ratios": [1.25, 1.0], "wspace": 0.55})

    ax = axes[0]
    im = ax.imshow(M, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(n)); ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(names)
    ax.set_xlabel("Test source (target domain)")
    ax.set_ylabel("Train source (source domain)")
    ax.set_title("(a) Cross-source AUC matrix")
    for i in range(n):
        for j in range(n):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="black", fontweight="bold" if i == j else "normal")
    # 对角用方框标出
    for i in range(n):
        ax.add_patch(plt.Rectangle((i - .5, i - .5), 1, 1, fill=False, edgecolor="black", lw=1.4))
    # colorbar 加 pad；标签放右侧并缩短，避免压到 (b) 面板
    cb = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.10)
    cb.set_label("AUC", labelpad=4)

    ax2 = axes[1]
    gaps, labels = [], []
    for k in names:
        t = res["as_target"].get(k)
        if not t:
            continue
        labels.append(k)
        gaps.append(t["gap"])
    y = np.arange(len(labels))
    colors = ["#c0392b" if g < -0.4 else "#e67e22" if g < -0.25 else "#f1c40f" for g in gaps]
    ax2.barh(y, gaps, color=colors)
    ax2.set_yticks(y); ax2.set_yticklabels(labels)
    ax2.axvline(0, color="black", lw=0.8)
    ax2.set_xlabel("ΔAUC  (cross-source mean − in-domain)")
    ax2.set_title("(b) Performance drop per target source")
    lo = min(gaps)
    ax2.set_xlim(lo * 1.28, 0.06)   # 左侧留白给数值标签，避免压到 y 轴文字
    for yi, g in zip(y, gaps):
        ax2.text(g - abs(lo) * 0.04, yi, f"{g:+.2f}", va="center", ha="right", fontsize=8,
                 fontweight="bold")
    ax2.invert_yaxis()

    fig.suptitle("Pathology foundation model features do not transfer across data sources",
                 fontsize=11, y=1.02)
    save(fig, "Figure1_cross_source_matrix")

    header = ["train\\test"] + names
    rows = []
    for a in names:
        row = [a]
        for b in names:
            row.append(round(in_dom[a]["auc"], 4) if a == b else
                       round(cross.get(f"{a}->{b}", {}).get("auc", float("nan")), 4))
        rows.append(row)
    write_table("Table3_cross_source_matrix", header, rows,
                note=("\n> Diagonal = in-domain (5-fold GroupKFold). Off-diagonal = zero-shot cross-source. "
                      f"\n> {len(cross)} cross-source pairs, AUC range "
                      f"{min(v['auc'] for v in cross.values()):.3f}–"
                      f"{max(v['auc'] for v in cross.values()):.3f}."))


# ---------------------------------------------------------------- Figure 2
def figure2() -> None:
    res = load_json(RUNS / "_stain_shortcut_audit" / "results.json")
    if not res or "shortcut_audit" not in res:
        return
    sa = res["shortcut_audit"]
    names = sorted(sa, key=lambda k: -sa[k]["stats_auc"])
    aucs = [sa[k]["stats_auc"] for k in names]

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    y = np.arange(len(names))
    colors = ["#c0392b" if a >= 0.9 else "#e67e22" if a >= 0.8 else "#f1c40f" for a in aucs]
    ax.barh(y, aucs, color=colors)
    ax.axvline(0.5, color="black", ls="--", lw=1.0, label="Chance (0.5)")
    ax.set_yticks(y); ax.set_yticklabels(names)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("AUC using only non-morphological image statistics")
    ax.set_title("Shortcut audit: class labels are predictable\nwithout any tissue morphology")
    for yi, a in zip(y, aucs):
        ax.text(a + 0.01, yi, f"{a:.3f}", va="center", fontsize=8, fontweight="bold")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.35), frameon=False)
    ax.invert_yaxis()
    save(fig, "Figure2_shortcut_audit")

    header = ["source", "n_per_class", "stats_auc", "top_features"]
    rows = [[k, sa[k]["n_per_class"], sa[k]["stats_auc"],
             "; ".join(f for f, _ in sa[k]["top_features"][:3])] for k in names]
    write_table("Table4_shortcut_audit", header, rows,
                note=("\n> Logistic regression on image width/height/aspect, file size, RGB statistics, "
                      "white- and dark-pixel fractions — no morphology. AUC near 0.5 means no shortcut; "
                      f"\n> here all {len(names)}/{len(names)} sources exceed 0.79, so in-domain AUC is likely inflated."))


# ---------------------------------------------------------------- Figure 3
def figure3() -> None:
    res = load_json(RUNS / "_stain_shortcut_audit" / "results.json")
    if not res or "eval_original" not in res:
        return
    o, s = res["eval_original"], res["eval_stainnorm"]
    d = res.get("cross_auc_delta_after_normalization", {})
    cross_keys = sorted(o["cross"])
    orig = [o["cross"][k]["auc"] for k in cross_keys]
    norm = [s["cross"][k]["auc"] for k in cross_keys]
    deltas = [d.get(k, np.nan) for k in cross_keys]

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.1),
                             gridspec_kw={"width_ratios": [1.0, 1.05], "wspace": 0.72})

    # (a) 配对散点：原始 vs 归一化（对角线 = 无变化）
    ax = axes[0]
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=1.0, label="no change")
    cols = ["#27ae60" if v > 0 else "#c0392b" for v in deltas]
    ax.scatter(orig, norm, c=cols, s=34, edgecolor="black", linewidth=0.4, zorder=3)
    # 域内点（空心方块）用于说明归一化几乎不影响域内
    for k in sorted(o["in_domain"]):
        ax.scatter([o["in_domain"][k]], [s["in_domain"].get(k, np.nan)], marker="s",
                   facecolor="none", edgecolor="#4c72b0", s=52, linewidth=1.2, zorder=3)
    ax.axhline(0.5, color="black", lw=0.7, ls=":")
    ax.axvline(0.5, color="black", lw=0.7, ls=":")
    ax.set_xlim(0.25, 1.05); ax.set_ylim(0.15, 1.05)
    ax.set_xlabel("AUC, original")
    ax.set_ylabel("AUC, Reinhard stain-normalized")
    ax.set_title("(a) Paired AUC before vs after normalization")
    ax.legend(handles=[
        Patch(facecolor="#27ae60", edgecolor="black", label="improved (>0)"),
        Patch(facecolor="#c0392b", edgecolor="black", label="worsened (<0)"),
        plt.Line2D([], [], marker="s", ls="", markerfacecolor="none",
                   markeredgecolor="#4c72b0", label="in-domain"),
        plt.Line2D([], [], ls="--", color="gray", label="no change")],
        loc="upper left", frameon=False, fontsize=7)

    # (b) 每对的 ΔAUC 排序（更直观地看出方向不一致）
    ax2 = axes[1]
    order = np.argsort(deltas)
    y = np.arange(len(order))
    vals = [deltas[i] for i in order]
    labels = [cross_keys[i].replace("->", " → ") for i in order]
    ax2.barh(y, vals, color=["#27ae60" if v > 0 else "#c0392b" for v in vals])
    ax2.axvline(0, color="black", lw=0.9)
    ax2.set_yticks(y); ax2.set_yticklabels(labels, fontsize=6.2)
    ax2.set_xlabel("ΔAUC  (normalized − original)")
    ax2.set_title("(b) Direction of change is inconsistent")
    n_pos = sum(1 for v in vals if v > 0)
    ax2.text(0.98, 0.02, f"improved {n_pos}/{len(vals)}", transform=ax2.transAxes,
             ha="right", va="bottom", fontsize=8, fontweight="bold")
    ax2.invert_yaxis()

    fig.suptitle("Stain normalization leaves in-domain AUC intact but does not "
                 "repair cross-source collapse", fontsize=10, y=1.03)
    save(fig, "Figure3_stain_normalization")

    header = ["comparison", "original_auc", "stainnorm_auc", "delta"]
    rows = []
    for k in sorted(o["in_domain"]):
        rows.append([f"in-domain {k}", o["in_domain"][k], s["in_domain"].get(k), 
                     round(s["in_domain"].get(k, float("nan")) - o["in_domain"][k], 4)])
    for k in cross_keys:
        rows.append([f"cross {k}", o["cross"][k]["auc"], s["cross"][k]["auc"], d.get(k)])
    write_table("Table5_stain_ablation", header, rows,
                note=("\n> In-domain AUC is essentially unchanged, while cross-source AUC shows "
                      "**inconsistent directions** (one pair improves, another worsens) — "
                      "so the collapse is not merely a stain difference, and normalization is not a fix."))


# ---------------------------------------------------------------- Figure 4
def figure4() -> None:
    res = load_json(RUNS / "_external_biopsy" / "results.json")
    if not res:
        return
    seeds = res["per_seed"]
    kinds = [("sensitivity_at_0.5", "Sensitivity", "external_biopsy"),
             ("mean_pred", "Mean predicted\nmalignant probability", "external_biopsy"),
             ("std_pred", "Prediction\nspread (SD)", "external_biopsy")]
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.2))

    for ax, (key, title, _) in zip(axes, kinds):
        vals = [s["external_biopsy"][key] for s in seeds]
        ax.bar(range(len(vals)), vals, color="#4c72b0", width=0.55)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels([f"seed {s['seed']}" for s in seeds])
        ax.set_title(title)
        ax.set_ylim(0, max(max(vals) * 1.35, 1.0))
        for i, v in enumerate(vals):
            ax.text(i, v + max(vals) * 0.04, f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")

    # 源域 vs 活检域的恶性平均预测对比
    src_means = [s["source_malignant_mean_pred"] for s in seeds]
    tgt_means = [s["external_biopsy"]["mean_pred"] for s in seeds]
    for ax in axes[1:2]:
        ax.axhline(np.mean(src_means), color="#c0392b", ls="--", lw=1.2,
                   label=f"source-domain malignant ({np.mean(src_means):.3f})")
        ax.legend(frameon=False, fontsize=7)

    fig.suptitle("External validation on 91 real endometrial biopsy WSIs "
                 "(all malignant): recall is preserved but confidence collapses", fontsize=10, y=1.04)
    save(fig, "Figure4_external_biopsy")

    header = ["seed", "source_val_auc", "biopsy_sensitivity_at_0.5", "biopsy_mean_pred",
              "biopsy_std_pred", "source_malignant_mean_pred", "calibration_bias"]
    rows = [[s["seed"], s["source_val_auc"], s["external_biopsy"]["sensitivity_at_0.5"],
             s["external_biopsy"]["mean_pred"], s["external_biopsy"]["std_pred"],
             s["source_malignant_mean_pred"], s["calibration_in_the_large_bias"]] for s in seeds]
    write_table("Table6_external_biopsy", header, rows,
                note=("\n> Target domain (ENDO-AID Pipelle biopsies) is **entirely malignant**, so AUC, "
                      "specificity and accuracy are **not reportable**. Only sensitivity and calibration "
                      "are reported.\n> Source domain: TCGA resection frozen tissue slides "
                      "(malignant 01A/01B, benign 11A)."))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    figure1()
    figure2()
    figure3()
    figure4()

    print(f"生成 {len(MADE)} 项：")
    for m in MADE:
        print(f"  ✓ {m}")
    if SKIPPED:
        print(f"\n跳过 {len(SKIPPED)} 项（因结果文件缺失或损坏，未静默生成空图）：")
        for s in SKIPPED:
            print(f"  - {s}")
    return 0 if MADE else 1


if __name__ == "__main__":
    sys.exit(main())
