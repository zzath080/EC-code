#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
16_match_uni2h_to_local.py
==========================
把 UNI2-h 特征清单与本地已下载切片对齐，算出「本次实际需要下载多少特征」。

关键问题：W8Yi 的 UNI2-h 特征覆盖 TCGA-UCEC 的 566 张**诊断切片**（01Z），
而 TCGA 的 54 张正常组织切片是**组织切片**（11A），两者可能不重叠。
本脚本给出确切的重叠数量，避免下载用不上的特征。

输出：
    data/manifests/uni2h/needed_features.csv       需要下载的特征（本地有对应 WSI）
    data/manifests/uni2h/match_report.json         对齐统计
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANI = ROOT / "data" / "manifests"
OUT = MANI / "uni2h"


def norm_id(name: str) -> str:
    """归一化切片标识，去掉扩展名与 GDC 文件名里的 UUID 段。

    GDC 侧：  TCGA-2E-A9G8-01Z-00-DX1.DCD3E31B-1774-40BA-805F-52FB9801D6C0.svs
    HF 侧：   TCGA-2E-A9G8-01Z-00-DX1.h5
    统一为：  TCGA-2E-A9G8-01Z-00-DX1
    """
    stem = re.sub(r"\.(svs|h5|tif|tiff|ndpi)$", "", name, flags=re.I)
    # 去掉 ".UUID"（8-4-4-4-12 十六进制）
    stem = re.sub(r"\.[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$", "", stem)
    return stem


def local_slide_ids() -> dict[str, dict]:
    """本地所有可用切片的 slide_id（含 sample_type）。"""
    out: dict[str, dict] = {}
    # 来自 slide_master（含 sample_type）
    mpath = MANI / "slide_master.csv"
    master = {}
    if mpath.exists():
        for r in csv.DictReader(open(mpath, encoding="utf-8-sig")):
            master[r["file_name"]] = r

    for d in [ROOT / "TCGA-UCEC", ROOT / "data" / "raw"]:
        if not d.exists():
            continue
        for ext in ("*.svs", "*.tif", "*.tiff"):
            for p in d.rglob(ext):
                sid = norm_id(p.name)
                meta = master.get(p.name, {})
                out[sid] = {
                    "slide_id": sid,
                    "file_name": p.name,
                    "local_path": str(p.relative_to(ROOT)),
                    "sample_type": meta.get("sample_type", "Unknown"),
                    "size": p.stat().st_size,
                }
    return out


def main() -> int:
    feat_csv = OUT / "TCGA-UCEC_features.csv"
    if not feat_csv.exists():
        print("[ERROR] 先跑 scripts/15_uni2h_manifest.py", file=sys.stderr)
        return 2
    feats = {norm_id(r["name"]): r for r in csv.DictReader(open(feat_csv, encoding="utf-8-sig"))}
    local = local_slide_ids()

    overlap = sorted(set(feats) & set(local))
    feat_only = sorted(set(feats) - set(local))
    local_only = sorted(set(local) - set(feats))

    by_type: dict[str, int] = {}
    for sid in overlap:
        t = local[sid]["sample_type"]
        by_type[t] = by_type.get(t, 0) + 1

    total_bytes = sum(int(feats[sid]["size"]) for sid in overlap)

    report = {
        "n_uni2h_features": len(feats),
        "n_local_slides": len(local),
        "n_overlap": len(overlap),
        "n_feature_only": len(feat_only),
        "n_local_only": len(local_only),
        "overlap_by_sample_type": by_type,
        "overlap_total_gb": round(total_bytes / 1024 ** 3, 2),
        "local_only_examples": [local[s]["file_name"] for s in local_only[:10]],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "match_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    with open(OUT / "needed_features.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["slide_id", "sample_type", "size", "local_path", "url"])
        for sid in overlap:
            w.writerow([sid, local[sid]["sample_type"], feats[sid]["size"],
                        local[sid]["local_path"], feats[sid]["url"]])

    print(json.dumps(report, indent=2, ensure_ascii=False))

    # 给出可用性判断
    n_norm = by_type.get("Solid Tissue Normal", 0)
    n_tum = by_type.get("Primary Tumor", 0) + by_type.get("Recurrent Tumor", 0)
    print(f"\n可立即使用的本地+特征配对：正常 {n_norm} 张，肿瘤 {n_tum} 张")
    if n_norm == 0:
        print("⚠️ 正常组织切片没有对应的公开 UNI2-h 特征（诊断切片与组织切片不重叠）")
        print("   → 若要跑通基线，需二选一：")
        print("      a) 下载 01Z 诊断切片（有公开特征，但无正常对照）")
        print("      b) 自行提取特征（需 torch + Phikon-v2/UNI2-h 权重）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
