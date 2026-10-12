#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
40_verify_patch_dataset.py
==========================
解压并核实 patch 级子宫内膜数据集的真实结构、类别分布与图像规格。

输出：
    data/manifests/endometrial_patch_summary.json
    data/manifests/endometrial_patch_files.csv
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="data/raw/endometrial_patch/endometrial-histopathology.zip")
    ap.add_argument("--extract", action="store_true")
    args = ap.parse_args()

    zpath = ROOT / args.zip
    if not zpath.exists():
        print(f"[ERROR] 缺 {zpath}", file=sys.stderr)
        return 2

    with zipfile.ZipFile(zpath) as z:
        names = z.namelist()
        print(f"zip 条目数：{len(names)}")

        images = [n for n in names if n.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))]
        print(f"图像文件：{len(images)}")

        # 类别 = 图像所在的一级/二级目录
        by_top = collections.Counter()
        by_pair = collections.Counter()
        for n in images:
            parts = n.split("/")
            if len(parts) >= 2:
                by_top[parts[0]] += 1
            if len(parts) >= 3:
                by_pair[f"{parts[0]}/{parts[1]}"] += 1

        print("\n一级目录分布：")
        for k, v in by_top.most_common():
            print(f"  {k:30s} {v}")
        if by_pair:
            print("\n二级目录分布（类别）：")
            for k, v in by_pair.most_common():
                print(f"  {k:45s} {v}")

        total_bytes = sum(i.file_size for i in z.infolist() if i.filename.lower().endswith(
            (".jpg", ".jpeg", ".png", ".tif", ".tiff")))
        summary = {
            "zip": str(zpath.relative_to(ROOT)),
            "n_entries": len(names),
            "n_images": len(images),
            "images_total_mb": round(total_bytes / 1024 ** 2, 1),
            "by_top_dir": dict(by_top.most_common()),
            "by_class_dir": dict(by_pair.most_common()),
            "classnames": sorted({k.split("/")[-1] for k in by_pair}) if by_pair else [],
        }

        out_dir = ROOT / "data" / "manifests"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "endometrial_patch_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        with open(out_dir / "endometrial_patch_files.csv", "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["path", "size_bytes"])
            for i in z.infolist():
                if i.filename.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
                    w.writerow([i.filename, i.file_size])

        if args.extract:
            dest = ROOT / "data" / "raw" / "endometrial_patch" / "extracted"
            dest.mkdir(parents=True, exist_ok=True)
            z.extractall(dest)
            print(f"\n已解压到 {dest}")

    print(f"\n-> {ROOT/'data'/'manifests'/'endometrial_patch_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
