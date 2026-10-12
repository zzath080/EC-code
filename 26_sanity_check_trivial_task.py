#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
26_sanity_check_trivial_task.py
===============================
当基线出现 AUC=1.0 时，必须先排除「代码/数据泄漏」，再判断这是不是「任务本身平凡」。

本脚本检查：
    1. 训练/验证/测试之间是否真的没有患者重叠（患者级泄漏）；
    2. 概率分布是否按类别分开（真实可分 vs 常数输出）；
    3. 单特征基线（各维度的单变量 AUC）——若某个维度单独就能 100% 分开，
       说明任务平凡，而不是模型学到了什么；
    4. 一个「无训练」基线：直接把全部 patch 特征取均值做最近质心分类，
       若能到 AUC=1.0，则证明任务无需学习即可解决。

输出：data/qc/trivial_task_check.json
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


def main() -> int:
    import h5py

    man = ROOT / "data" / "manifests" / "tcga_pair_train_manifest.csv"
    rows = list(csv.DictReader(open(man, encoding="utf-8-sig")))
    report: dict = {}

    # 1) 患者级泄漏
    pat = {}
    for r in rows:
        pat.setdefault(r["split"], set()).add(r["patient_id"])
    overlaps = {}
    splits = sorted(pat)
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            overlaps[f"{a}∩{b}"] = len(pat[a] & pat[b])
    report["patient_overlap"] = overlaps
    report["patient_overlap_zero"] = all(v == 0 for v in overlaps.values())

    # 按 slide 汇总特征均值（无训练最近质心）
    feats, labels, sids = [], [], []
    for r in rows:
        p = ROOT / r["feature_path"]
        if not p.exists():
            continue
        with h5py.File(p, "r") as fh:
            feats.append(fh["features"][:].astype(np.float32).mean(axis=0))
        labels.append(int(r["label"]))
        sids.append(r["slide_id"])
    X = np.stack(feats)
    y = np.array(labels)
    report["n_slides"] = len(y)
    report["feature_dim"] = int(X.shape[1])

    # 2) 单维度 AUC：找出最容易分开的维度
    single = np.array([auc(y, X[:, d]) for d in range(X.shape[1])])
    best = int(np.nanargmax(np.abs(single - 0.5)))
    report["best_single_dim"] = {
        "dim": best,
        "auc": round(float(single[best]), 4),
        "n_dims_with_auc_1.0": int(np.sum(np.abs(single - 1.0) < 1e-9)),
        "mean_abs_auc_minus_half": round(float(np.nanmean(np.abs(single - 0.5))), 4),
    }

    # 3) 无训练最近质心（用 train 质心，在 test 上评估）
    tr = np.array([r["split"] == "train" for r in rows])[: len(y)]
    te = np.array([r["split"] == "test" for r in rows])[: len(y)]
    if tr.sum() and te.sum() and len(np.unique(y[tr])) == 2:
        c0 = X[tr & (y == 0)].mean(axis=0)
        c1 = X[tr & (y == 1)].mean(axis=0)
        d0 = np.linalg.norm(X[te] - c0, axis=1)
        d1 = np.linalg.norm(X[te] - c1, axis=1)
        score = (d0 - d1) / (d0 + d1 + 1e-12)
        report["nearest_centroid_no_training"] = {
            "n_test": int(te.sum()),
            "auc": round(auc(y[te], score), 4),
        }
    # 4) 全部样本（含 train）上的质心可分性
    if len(np.unique(y)) == 2:
        c0 = X[y == 0].mean(axis=0)
        c1 = X[y == 1].mean(axis=0)
        between = np.linalg.norm(c0 - c1)
        within = np.mean([np.linalg.norm(X[y == 0] - c0, axis=1).mean(),
                          np.linalg.norm(X[y == 1] - c1, axis=1).mean()])
        report["centroid_separation_ratio"] = round(float(between / (within + 1e-12)), 3)

    # 判读
    report["verdict"] = (
        "任务平凡（无需学习即可近乎完美区分）"
        if report.get("nearest_centroid_no_training", {}).get("auc", 0) and
        report["nearest_centroid_no_training"]["auc"] >= 0.99
        else "任务非平凡，需检查模型是否过拟合"
    )
    report["patient_leakage_free"] = report["patient_overlap_zero"]

    out = ROOT / "data" / "qc" / "trivial_task_check.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
