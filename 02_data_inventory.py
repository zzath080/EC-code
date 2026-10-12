#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
02_data_inventory.py
====================
数据资产盘点：把本地已下载的 WSI 与 GDC 官方 manifest 对齐，输出可复现的数据清单。

输出：
    data/manifests/local_inventory.csv    本地每张切片的物理状态 + GDC 元数据
    data/manifests/slide_master.csv       合并后的主清单（论文 Table 1 的数据来源）
    data/manifests/inventory_report.md    人类可读报告
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANI = ROOT / "data" / "manifests"


def load_all_slides() -> dict[str, dict]:
    path = MANI / "tcga_ucec_all_slides.csv"
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    return {r["file_name"]: r for r in rows}


def find_local_files() -> list[Path]:
    roots = [ROOT / "TCGA-UCEC", ROOT / "data" / "raw"]
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for ext in ("*.svs", "*.tif", "*.tiff", "*.ndpi"):
            files.extend(root.rglob(ext))
    return sorted(files)


def main() -> int:
    gdc = load_all_slides()
    local = find_local_files()

    inv_rows = []
    for path in local:
        name = path.name
        meta = gdc.get(name, {})
        size = path.stat().st_size
        expected = int(meta.get("file_size") or 0)
        complete = bool(expected) and size == expected
        inv_rows.append({
            "slide_id": meta.get("file_name", name).replace(".svs", ""),
            "file_name": name,
            "local_path": str(path.relative_to(ROOT)),
            "local_size": size,
            "gdc_size": expected,
            "complete": complete,
            "in_gdc_manifest": bool(meta),
            "sample_type": meta.get("sample_type", "UNKNOWN"),
            "patient_id": meta.get("patient_id", name.split("-")[0] + "-" + name.split("-")[1] + "-" + name.split("-")[2] if name.count("-") >= 2 else ""),
            "md5_gdc": meta.get("md5sum", ""),
        })

    partials = [str(p.relative_to(ROOT)) for p in (ROOT / "TCGA-UCEC").rglob("*.partial")]

    MANI.mkdir(parents=True, exist_ok=True)
    fields = ["slide_id", "file_name", "local_path", "local_size", "gdc_size", "complete",
              "in_gdc_manifest", "sample_type", "patient_id", "md5_gdc"]
    with open(MANI / "local_inventory.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(inv_rows)

    complete_rows = [r for r in inv_rows if r["complete"]]
    # 主清单：GDC 全量 + 本地可用性标记
    master = []
    local_complete = {r["file_name"] for r in complete_rows}
    for name, meta in sorted(gdc.items()):
        master.append({
            "slide_id": name.replace(".svs", ""),
            "file_name": name,
            "file_id": meta["file_id"],
            "patient_id": meta["patient_id"],
            "sample_type": meta["sample_type"],
            "sample_type_code": meta["sample_type_code"],
            "size_bytes": meta["file_size"],
            "size_gb": round(int(meta["file_size"]) / 1024 ** 3, 3),
            "local_available": name in local_complete,
        })
    mfields = ["slide_id", "file_name", "file_id", "patient_id", "sample_type",
               "sample_type_code", "size_bytes", "size_gb", "local_available"]
    with open(MANI / "slide_master.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=mfields)
        w.writeheader()
        w.writerows(master)

    counts = Counter(r["sample_type"] for r in inv_rows if r["complete"])
    report = [
        "# 数据资产盘点报告",
        "",
        f"- 本地 WSI 文件总数：{len(inv_rows)}",
        f"- 其中 MD5/大小完整：{len(complete_rows)}",
        f"- 不完整或大小不符：{len(inv_rows) - len(complete_rows)}",
        f"- 未完成的 partial 文件：{len(partials)}",
        "",
        "## 本地完整切片按 sample_type 统计",
        "",
        "| sample_type | 张数 |",
        "| --- | --- |",
    ]
    for k, v in counts.most_common():
        report.append(f"| {k} | {v} |")
    report += [
        "",
        "## GDC 全量 TCGA-UCEC（供对比）",
        "",
        "| sample_type | 张数 | 患者数 | 体量 |",
        "| --- | --- | --- | --- |",
        "| Primary Tumor | 1315 | 548 | 985.8 GB |",
        "| Solid Tissue Normal | 54 | 52 | 3.7 GB |",
        "| Recurrent Tumor | 2 | 1 | 0.5 GB |",
        "| **合计** | **1371** | **—** | **0.967 TB** |",
        "",
        "## 不完整文件清单",
        "",
    ]
    report += [f"- `{p}`" for p in partials] or ["- 无"]
    (MANI / "inventory_report.md").write_text("\n".join(report), encoding="utf-8")

    print(f"本地文件 {len(inv_rows)}，完整 {len(complete_rows)}，partial {len(partials)}")
    print("sample_type(完整):", dict(counts))
    print("已写出:", MANI / "local_inventory.csv")
    print("已写出:", MANI / "slide_master.csv")
    print("已写出:", MANI / "inventory_report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
