#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
25_check_id_consistency.py
==========================
端到端 ID 一致性检查：划分清单 ↔ patch 索引 ↔ 特征文件。

在跑完整训练前先做这一步，可以避免"跑了半小时才发现 ID 对不上"。
它检查每条链路的两端是否使用完全相同的 slide_id 字符串。

输出：data/qc/id_consistency.json
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QC = ROOT / "data" / "qc"


def load_split_ids() -> list[str]:
    p = ROOT / "data" / "manifests" / "tcga_pair_split_manifest.csv"
    if not p.exists():
        return []
    return [r["slide_id"] for r in csv.DictReader(open(p, encoding="utf-8-sig"))]


def main() -> int:
    report: dict = {}
    split_ids = set(load_split_ids())
    patch_ids = {p.stem for p in (ROOT / "data" / "patches").rglob("*.csv")}
    all_feat_ids = {p.stem for p in (ROOT / "data" / "features").rglob("*.h5")}
    # 公开预提取特征（uni2h_public）本来就没有本地 patch，属预期差异，单独统计
    public_ids = {p.stem for p in (ROOT / "data" / "features" / "uni2h_public").rglob("*.h5")}
    feat_ids = all_feat_ids - public_ids

    report["counts"] = {
        "split_manifest": len(split_ids),
        "patch_index": len(patch_ids),
        "own_features": len(feat_ids),
        "public_pretrained_features": len(public_ids),
    }
    report["split_vs_patch"] = {
        "missing_patch_for_split": sorted(split_ids - patch_ids)[:20],
        "n_missing": len(split_ids - patch_ids),
        "extra_patch_not_in_split": len(patch_ids - split_ids),
    }
    report["patch_vs_feature"] = {
        "patch_without_feature_yet": len(patch_ids - feat_ids),
        "feature_without_patch": sorted(feat_ids - patch_ids)[:20],
        "n_feature_without_patch": len(feat_ids - patch_ids),
    }
    report["split_vs_feature"] = {
        "split_with_feature": len(split_ids & feat_ids),
        "split_without_feature_yet": len(split_ids - feat_ids),
    }
    # 特征文件与划分清单直接配对后的标签分布
    pairs = split_ids & feat_ids
    labels = Counter()
    if pairs:
        for r in csv.DictReader(open(ROOT / "data" / "manifests" / "tcga_pair_split_manifest.csv",
                                     encoding="utf-8-sig")):
            if r["slide_id"] in pairs:
                labels[(r["split"], r["label"])] += 1
    report["paired_label_split_distribution"] = {f"{k[0]}/label={k[1]}": v for k, v in sorted(labels.items())}

    ok = (report["split_vs_patch"]["n_missing"] == 0
          and report["patch_vs_feature"]["n_feature_without_patch"] == 0)
    report["id_consistency_ok"] = ok

    QC.mkdir(parents=True, exist_ok=True)
    (QC / "id_consistency.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nID 一致性：{'✅ 通过' if ok else '❌ 存在问题'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
