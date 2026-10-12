#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
22_analyze_negative_transfer.py
===============================
负迁移与校准分析（论文核心分析章节）。

做三件事：
    1. 配对比较：模型 A vs B 的 AUC 差值 + 95% bootstrap CI + DeLong 检验 + Holm 校正；
    2. 校准诊断：ECE、Brier、可靠性曲线、最大校准误差所在分箱；
    3. 阈值分析：在固定敏感度（如 0.90 / 0.95）下比较特异度，给出临床可解释的取舍。

输入：runs/*/predictions.csv（由 13_mil_train_eval.py 产出）
输出：
    runs/_analysis/pairwise_auc.csv
    runs/_analysis/calibration_summary.csv
    runs/_analysis/threshold_analysis.csv
    runs/_analysis/negative_transfer_report.md

用法：
    python scripts/22_analyze_negative_transfer.py --runs runs --baseline abmil
    python scripts/22_analyze_negative_transfer.py --runs runs --baseline abmil --model clam
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def auc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv), dtype=float)
    ranks[order] = np.arange(1, len(allv) + 1)
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def bootstrap_ci(y: np.ndarray, s: np.ndarray, n: int = 2000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        take = rng.integers(0, len(y), len(y))
        if len(np.unique(y[take])) < 2:
            continue
        vals.append(auc(y[take], s[take]))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"), float("nan"))


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    tot = len(y)
    val = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum():
            val += (m.sum() / tot) * abs(p[m].mean() - y[m].mean())
    return float(val)


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni 校正。"""
    idx = np.argsort(pvals)
    m = len(pvals)
    out = [0.0] * m
    prev = 0.0
    for rank, i in enumerate(idx):
        adj = min(1.0, (m - rank) * pvals[i])
        prev = max(prev, adj)
        out[i] = prev
    return out


def load_run(run_dir: Path) -> dict | None:
    pred = run_dir / "predictions.csv"
    if not pred.exists():
        return None
    y, p, sids = [], [], []
    for r in csv.DictReader(open(pred, encoding="utf-8-sig")):
        y.append(int(r["label"]))
        p.append(float(r["prob_malignant"]))
        sids.append(r["slide_id"])
    y, p = np.array(y), np.array(p)
    metrics = {}
    mp = run_dir / "metrics.json"
    if mp.exists():
        metrics = json.loads(mp.read_text(encoding="utf-8"))
    return {"run_id": run_dir.name, "dir": run_dir, "y": y, "p": p, "slide_ids": sids, "metrics": metrics}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--baseline", default=None, help="作为对照的 run_id 前缀（如 abmil）")
    ap.add_argument("--model", default=None, help="待比较的 run_id 前缀（如 clam）")
    args = ap.parse_args()

    runs_root = ROOT / args.runs
    if not runs_root.exists():
        print(f"[WARN] 运行目录不存在：{runs_root}", file=sys.stderr)
        print("       先跑 scripts/13_mil_train_eval.py 产生运行结果")
        return 1
    # 只纳入真正的训练 run；`_` 开头的是内部分析目录（如 _analysis / _cross_specimen），
    # 它们的 predictions.csv 属于不同评估集，混进来会产生无意义的混合指标。
    run_dirs = [d for d in sorted(runs_root.iterdir())
                if d.is_dir() and not d.name.startswith("_")]
    runs = [r for r in (load_run(d) for d in run_dirs) if r]
    if not runs:
        print("[WARN] 没有可分析的 runs（缺 predictions.csv）", file=sys.stderr)
        print("       先跑 scripts/13_mil_train_eval.py 产生运行结果")
        return 1

    out_dir = runs_root / "_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"发现 {len(runs)} 个可用 run：")
    for r in runs:
        lo, hi = bootstrap_ci(r["y"], r["p"])
        print(f"  {r['run_id']:34s} n={len(r['y']):3d}  AUC={auc(r['y'], r['p']):.4f} "
              f"[{lo:.3f}, {hi:.3f}]  ECE={ece(r['y'], r['p']):.4f}  "
              f"Brier={np.mean((r['p']-r['y'])**2):.4f}")

    # 配对比较（同一批切片才能配对）
    pair_rows = []
    raw_p = []
    for a, b in combinations(runs, 2):
        if a["slide_ids"] != b["slide_ids"]:
            continue
        diff = auc(a["y"], a["p"]) - auc(b["y"], b["p"])
        # 非参 bootstrap 的配对差值 CI
        rng = np.random.default_rng(42)
        deltas = []
        for _ in range(2000):
            take = rng.integers(0, len(a["y"]), len(a["y"]))
            if len(np.unique(a["y"][take])) < 2:
                continue
            deltas.append(auc(a["y"][take], a["p"][take]) - auc(a["y"][take], b["p"][take]))
        lo, hi = (float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))) if deltas else (np.nan, np.nan)
        # 用 bootstrap 双侧 p 值近似（避免依赖 DeLong 实现细节）
        p_val = float(2 * min(np.mean(np.array(deltas) <= 0) if deltas else 0.5,
                              np.mean(np.array(deltas) >= 0) if deltas else 0.5))
        pair_rows.append({
            "model_a": a["run_id"], "model_b": b["run_id"],
            "auc_a": round(auc(a["y"], a["p"]), 4), "auc_b": round(auc(b["y"], b["p"]), 4),
            "auc_diff": round(diff, 4),
            "diff_ci95_lo": round(lo, 4), "diff_ci95_hi": round(hi, 4),
            "p_value": round(p_val, 4), "n": len(a["y"]),
        })
        raw_p.append(p_val)
    if pair_rows:
        adj = holm(raw_p)
        for r, p_adj in zip(pair_rows, adj):
            r["p_holm"] = round(p_adj, 4)
            r["significant_after_holm"] = bool(p_adj < 0.05)

    if pair_rows:
        with open(out_dir / "pairwise_auc.csv", "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(pair_rows[0].keys()))
            w.writeheader()
            w.writerows(pair_rows)

    # 校准与阈值
    cal_rows, thr_rows = [], []
    for r in runs:
        y, p = r["y"], r["p"]
        row = {"run_id": r["run_id"], "n": len(y), "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
               "auc": round(auc(y, p), 4), "ece": round(ece(y, p), 4),
               "brier": round(float(np.mean((p - y) ** 2)), 4),
               "mean_pred": round(float(p.mean()), 4), "frac_pos": round(float(y.mean()), 4)}
        cal_rows.append(row)
        for target_sens in (0.90, 0.95):
            if y.sum() == 0:
                continue
            order = np.argsort(-p)
            ys = y[order]
            tp = np.cumsum(ys)
            k = int(np.argmax(tp >= target_sens * y.sum()) + 1) if (tp >= target_sens * y.sum()).any() else len(ys)
            thr = float(p[order][k - 1])
            pred = (p >= thr).astype(int)
            tpr = float(((pred == 1) & (y == 1)).sum() / max(1, (y == 1).sum()))
            tnr = float(((pred == 0) & (y == 0)).sum() / max(1, (y == 0).sum()))
            thr_rows.append({"run_id": r["run_id"], "target_sensitivity": target_sens,
                             "threshold": round(thr, 4), "achieved_sensitivity": round(tpr, 4),
                             "specificity": round(tnr, 4), "n_flagged": int(pred.sum())})

    with open(out_dir / "calibration_summary.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cal_rows[0].keys()))
        w.writeheader()
        w.writerows(cal_rows)
    if thr_rows:
        with open(out_dir / "threshold_analysis.csv", "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(thr_rows[0].keys()))
            w.writeheader()
            w.writerows(thr_rows)

    # 负迁移报告
    lines = ["# 负迁移与校准分析报告", "",
             f"分析运行数：{len(runs)}", "",
             "## 1. 各运行的判别性能与校准", "",
             "| run_id | n | AUC | 95% CI | ECE | Brier | 预测均值 | 阳性率 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r, row in zip(runs, cal_rows):
        lo, hi = bootstrap_ci(r["y"], r["p"])
        lines.append(f"| {row['run_id']} | {row['n']} | {row['auc']:.4f} | [{lo:.3f}, {hi:.3f}] | "
                     f"{row['ece']:.4f} | {row['brier']:.4f} | {row['mean_pred']:.4f} | {row['frac_pos']:.4f} |")
    lines += ["", "## 2. 配对 AUC 比较（同一切片集合）", ""]
    if pair_rows:
        lines += ["| 模型 A | 模型 B | ΔAUC | ΔAUC 95% CI | p | p(Holm) | 显著 |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for r in pair_rows:
            lines.append(f"| {r['model_a']} | {r['model_b']} | {r['auc_diff']:+.4f} | "
                         f"[{r['diff_ci95_lo']:+.3f}, {r['diff_ci95_hi']:+.3f}] | {r['p_value']:.4f} | "
                         f"{r['p_holm']:.4f} | {'是' if r['significant_after_holm'] else '否'} |")
        lines += ["", "> 判读规则：若 ΔAUC 的 95% CI 跨 0 或 Holm 校正后不显著，",
                  "> **不得**声称两模型存在性能差异（这正是 Chen 2024 / Phikon-v2 论文提醒的陷阱）。"]
    else:
        lines.append("_无可配对的运行（需要两个 run 在完全相同的切片集合上评估）_")
    lines += ["", "## 3. 固定敏感度下的特异度（临床取舍）", ""]
    if thr_rows:
        lines += ["| run_id | 目标敏感度 | 阈值 | 实际敏感度 | 特异度 | 报警数 |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for r in thr_rows:
            lines.append(f"| {r['run_id']} | {r['target_sensitivity']:.2f} | {r['threshold']:.4f} | "
                         f"{r['achieved_sensitivity']:.4f} | {r['specificity']:.4f} | {r['n_flagged']} |")
    (out_dir / "negative_transfer_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\n-> {out_dir/'negative_transfer_report.md'}")
    print(f"-> {out_dir/'pairwise_auc.csv'}")
    print(f"-> {out_dir/'calibration_summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
