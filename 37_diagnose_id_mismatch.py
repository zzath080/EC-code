#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
37_diagnose_id_mismatch.py
==========================
当「特征可用且有标签的切片 = 0」时，用来定位是哪两套 ID 对不上。

输出：两边各自的前若干样例 ID、共同前缀、以及规范化后能否匹配。
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UUID_RE = re.compile(r"\.[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$")


def canon(s: str) -> str:
    return UUID_RE.sub("", s)


def main() -> int:
    split_p = ROOT / "data" / "manifests" / "tcga_pair_split_manifest.csv"
    rows = list(csv.DictReader(open(split_p, encoding="utf-8-sig")))
    split_ids = [r["slide_id"] for r in rows]

    feats = [p.stem for p in (ROOT / "data" / "features").rglob("*.h5") if "_flat" not in str(p)]
    feats = sorted(set(feats))

    print(f"划分清单 {len(split_ids)} 条，特征文件 {len(feats)} 个\n")
    print("划分清单样例：")
    for s in split_ids[:5]:
        print(f"  {s!r}")
    print("\n特征文件样例：")
    for s in feats[:5]:
        print(f"  {s!r}")

    A, B = set(split_ids), set(feats)
    print(f"\n直接交集：{len(A & B)}")
    print(f"规范化后交集：{len({canon(x) for x in A} & {canon(x) for x in B})}")
    print(f"划分清单含 UUID 的条数：{sum(1 for x in A if UUID_RE.search(x))}")
    print(f"特征名含 UUID 的个数：{sum(1 for x in B if UUID_RE.search(x))}")

    only_split = sorted(A - B)[:5]
    only_feat = sorted(B - A)[:5]
    print(f"\n仅在划分清单中（前 5）："
          f"\n  " + "\n  ".join(only_split))
    print(f"\n仅在特征中（前 5）："
          f"\n  " + "\n  ".join(only_feat))

    # 给出修复建议
    if len({canon(x) for x in A} & {canon(x) for x in B}) > 0:
        print("\n→ 修复建议：两边对 slide_id 的规范化不一致。"
              "\n  统一规则：去掉扩展名与 GDC 文件名的 UUID 段（见 10_tiling.canonical_slide_id）。")
    else:
        print("\n→ 两边 ID 完全不同，需检查是否扫描到了不同的目录。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
