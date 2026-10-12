#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
20_merge_feature_manifest.py
============================
把分队列提取的特征（data/features/<encoder>/<cohort>/<slide_id>.h5）汇总为训练可用的
扁平目录 + 带标签的清单，供 13_mil_train_eval.py 使用。

产物：
    data/features/<encoder>/_flat/<slide_id>.h5      软链接或复制的扁平特征
    data/manifests/<cohort>_train_manifest.csv       含 slide_id,label,split,feature_path

用法：
    python scripts/20_merge_feature_manifest.py --cohort tcga_pair
    python scripts/20_merge_feature_manifest.py --cohort tcga_pair --copy   # 复制而非硬链接
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def link_or_copy(src: Path, dst: Path, copy: bool) -> str:
    if dst.exists():
        return "exists"
    try:
        if copy:
            shutil.copy2(src, dst)
            return "copied"
        os.link(src, dst)   # 硬链接，同盘零成本
        return "linked"
    except OSError:
        shutil.copy2(src, dst)
        return "copied"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="tcga_pair")
    ap.add_argument("--encoder", default="phikon-v2")
    ap.add_argument("--copy", action="store_true")
    args = ap.parse_args()

    split_csv = ROOT / "data" / "manifests" / f"{args.cohort}_split_manifest.csv"
    if not split_csv.exists():
        print(f"[ERROR] 缺少划分清单 {split_csv}，先跑 scripts/14_build_split_manifest.py", file=sys.stderr)
        return 2

    feat_root = ROOT / "data" / "features" / args.encoder
    flat = feat_root / "_flat"
    flat.mkdir(parents=True, exist_ok=True)

    # 收集所有队列子目录下的特征
    found: dict[str, Path] = {}
    for sub in feat_root.iterdir():
        if not sub.is_dir() or sub.name == "_flat":
            continue
        for h5 in sub.glob("*.h5"):
            found.setdefault(h5.stem, h5)
    # 公开 UNI2-h 特征也纳入（若存在）
    for h5 in (ROOT / "data" / "features" / "uni2h_public").rglob("*.h5"):
        found.setdefault(h5.stem, h5)

    rows = list(csv.DictReader(open(split_csv, encoding="utf-8-sig")))
    out_rows = []
    n_linked = 0
    for r in rows:
        sid = r["slide_id"]
        src = found.get(sid)
        if src is None:
            continue
        dst = flat / f"{sid}.h5"
        action = link_or_copy(src, dst, args.copy)
        n_linked += 1
        out_rows.append({
            "slide_id": sid,
            "patient_id": r.get("patient_id", ""),
            "label": r.get("label", ""),
            "split": r.get("split", ""),
            "sample_type": r.get("sample_type", ""),
            "feature_path": str(dst.relative_to(ROOT)),
            "feature_source": str(src.relative_to(ROOT)),
        })

    out_csv = ROOT / "data" / "manifests" / f"{args.cohort}_train_manifest.csv"
    fields = ["slide_id", "patient_id", "label", "split", "sample_type", "feature_path", "feature_source"]
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    from collections import Counter
    by_split = Counter((r["split"], r["label"]) for r in out_rows)
    print(f"特征可用且有标签的切片：{len(out_rows)}（来自 {len(found)} 个特征文件）")
    for (sp, lab), n in sorted(by_split.items()):
        print(f"  {sp:5s} label={lab}  {n} 张")
    print(f"扁平特征目录 -> {flat}")
    print(f"训练清单 -> {out_csv}")
    if not out_rows:
        print("\n[WARN] 尚无任何切片同时具备「特征」与「标签」；"
              "需先完成特征提取（scripts/12_extract_features.py --save）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
