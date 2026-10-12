#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
29_status_report.py
===================
项目数据状态总览：把「清单要求的量」与「本地实际拥有的量」对齐，指出缺口。

用途：每次接手项目时先跑这个，避免基于过时认知做决策。

输出：
    data/manifests/status_report.json
    data/manifests/状态总览.md
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANI = ROOT / "data" / "manifests"

WSI_EXTS = ("*.svs", "*.tif", "*.tiff", "*.ndpi")


def sid_of(name: str) -> str:
    s = re.sub(r"\.(svs|h5|tif|tiff|ndpi)$", "", name, flags=re.I)
    return re.sub(r"\.[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$", "", s)


def local_slides() -> dict[str, Path]:
    out: dict[str, Path] = {}
    for d in [ROOT / "TCGA-UCEC", ROOT / "data" / "raw"]:
        if not d.exists():
            continue
        for ext in WSI_EXTS:
            for p in d.rglob(ext):
                out.setdefault(sid_of(p.name), p)
    return out


def patch_ids() -> set[str]:
    """patch 索引的 slide_id 集合（文件名即规范 id）。"""
    return {p.stem for p in (ROOT / "data" / "patches").rglob("*.csv")}


def feature_ids() -> tuple[set[str], set[str]]:
    """返回 (自有特征 id 集, 公开预提取特征 id 集)，文件名即规范 id。"""
    fdir = ROOT / "data" / "features"
    public: set[str] = set()
    own: set[str] = set()
    for p in fdir.rglob("*.h5"):
        s = str(p)
        if "_flat" in s:
            continue
        (public if "uni2h_public" in s else own).add(p.stem)
    return own, public


def manifest_codes() -> Counter:
    """GDC 全量 manifest 的条码分布。"""
    p = MANI / "tcga_ucec_all_slides.csv"
    c: Counter = Counter()
    if p.exists():
        for r in csv.DictReader(open(p, encoding="utf-8-sig")):
            c[r.get("sample_type_code", "?")] += 1
    return c


def main() -> int:
    local = local_slides()
    patches = patch_ids()
    feats, public = feature_ids()
    codes = manifest_codes()

    local_codes = Counter()
    for sid in local:
        parts = sid.split("-")
        local_codes[parts[3] if len(parts) > 3 else "?"] += 1

    report = {
        "gdc_manifest_by_code": dict(codes.most_common()),
        "local_by_code": dict(local_codes.most_common()),
        "n_local_slides": len(local),
        "n_patch_index": len(patches),
        "n_own_features": len(feats),
        "n_public_features": len(public),
        "gap_patch_missing": sorted(set(local) - patches)[:30],
        "n_gap_patch_missing": len(set(local) - patches),
        "gap_feature_missing": sorted(patches - feats)[:30],
        "n_gap_feature_missing": len(patches - feats),
    }

    lines = [
        "# 项目数据状态总览",
        "",
        f"- GDC 全量清单：{sum(codes.values())} 张切片",
        f"- 本地切片：**{len(local)}** 张（{sum(local_codes.values())}）",
        f"- 已建 patch 索引：**{len(patches)}** 张",
        f"- 已提取自有特征：**{len(feats)}** 张",
        f"- 公开预提取特征：{len(public)} 张",
        "",
        "## 按条码分布（GDC 全量 vs 本地）",
        "",
        "| 条码 | GDC 全量 | 本地 |",
        "| --- | --- | --- |",
    ]
    for code in sorted(set(codes) | set(local_codes)):
        lines.append(f"| {code} | {codes.get(code, 0)} | {local_codes.get(code, 0)} |")
    lines += [
        "",
        f"## 缺口",
        "",
        f"- 有切片但**缺 patch 索引**：{report['n_gap_patch_missing']} 张",
        f"- 有 patch 索引但**缺特征**：{report['n_gap_feature_missing']} 张",
        "",
        "> 条码含义：`01Z`=原发瘤诊断切片(FFPE)，`01A/01B`=肿瘤组织切片(冰冻/TSA)，",
        "> `11A`=实性正常组织切片(冰冻)，`02Z`=复发瘤诊断切片。",
    ]
    (MANI / "状态总览.md").write_text("\n".join(lines), encoding="utf-8")
    (MANI / "status_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({k: v for k, v in report.items() if not k.startswith("gap_")}, indent=2, ensure_ascii=False))
    print("\n缺口：")
    print(f"  缺 patch 索引 {report['n_gap_patch_missing']} 张")
    print(f"  缺特征      {report['n_gap_feature_missing']} 张")
    print(f"\n-> {MANI/'状态总览.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
