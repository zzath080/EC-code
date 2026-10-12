#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
00_build_gdc_manifest.py
========================
从 GDC API 拉取 TCGA-UCEC 全部诊断切片（Slide Image）的完整元数据，
构建可复现的本地 manifest。

输出（写入 data/manifests/）：
    tcga_ucec_all_slides.csv      所有 Slide Image，含 sample_type / patient / 大小
    tcga_ucec_normals.csv         仅 Solid Tissue Normal（负类候选）
    tcga_ucec_tumor.csv           仅 Primary Tumor / Recurrent Tumor（正类候选）
    gdc_manifest_normals.txt      gdc-client 可直接使用的 manifest（正常组织）
    gdc_manifest_tumor.txt        gdc-client 可直接使用的 manifest（肿瘤）
    tcga_ucec_summary.json        分组统计

用法：
    python scripts/00_build_gdc_manifest.py
    python scripts/00_build_gdc_manifest.py --project TCGA-UCEC

设计说明：
    - 只读 GDC API，不下载切片本体。
    - sample_type 是 TCGA 条码第 4 段（01Z=原发瘤, 11Z=实性正常组织, 02Z=复发瘤）。
    - 显式记录 sample_type_code，便于后续按条码规则复核。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
GDC_FIELDS = [
    "file_id",
    "file_name",
    "file_size",
    "md5sum",
    "state",
    "access",
    "data_category",
    "data_type",
    "experimental_strategy",
    "platform",
    "cases.submitter_id",
    "cases.samples.sample_type",
    "cases.samples.submitter_id",
    "cases.samples.portions.analytes.aliquots.aliquot_id",
    "cases.disease_type",
    "cases.primary_site",
]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def query_gdc_slides(project: str, page_size: int = 500) -> list[dict]:
    """分页拉取指定 project 的所有 Slide Image 文件记录。"""
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": [project]}},
            {"op": "in", "content": {"field": "data_category", "value": ["Biospecimen"]}},
            {"op": "in", "content": {"field": "data_type", "value": ["Slide Image"]}},
        ],
    }
    hits: list[dict] = []
    frm = 0
    while True:
        payload = {
            "filters": filters,
            "fields": ",".join(GDC_FIELDS),
            "format": "JSON",
            "size": page_size,
            "from": frm,
        }
        req = urllib.request.Request(
            GDC_FILES_ENDPOINT,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.load(resp)["data"]
        batch = data["hits"]
        hits.extend(batch)
        total = data["pagination"]["total"]
        print(f"  fetched {len(hits)}/{total}", flush=True)
        frm += page_size
        if len(hits) >= total or not batch:
            break
        time.sleep(0.3)
    return hits


def flatten(hit: dict) -> dict:
    """把嵌套 GDC 记录压平成一行。"""
    case = (hit.get("cases") or [{}])[0]
    sample = (case.get("samples") or [{}])[0]
    file_name = hit.get("file_name") or ""
    parts = file_name.split("-")
    sample_type_code = parts[3] if len(parts) > 3 else ""
    patient = case.get("submitter_id") or "-".join(parts[:3])
    return {
        "file_id": hit.get("file_id", ""),
        "file_name": file_name,
        "file_size": hit.get("file_size", 0),
        "md5sum": hit.get("md5sum", ""),
        "state": hit.get("state", ""),
        "access": hit.get("access", ""),
        "platform": hit.get("platform", ""),
        "patient_id": patient,
        "sample_id": sample.get("submitter_id", ""),
        "sample_type": sample.get("sample_type", ""),
        "sample_type_code": sample_type_code,
        "disease_type": case.get("disease_type", ""),
        "primary_site": case.get("primary_site", ""),
    }


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def write_gdc_manifest(path: Path, rows: list[dict]) -> None:
    """gdc-client 格式 manifest：id / filename / md5 / size / state。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["id", "filename", "md5", "size", "state"])
        for row in rows:
            writer.writerow([row["file_id"], row["file_name"], row["md5sum"], row["file_size"], row["state"]])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="TCGA-UCEC")
    args = parser.parse_args()

    root = repo_root()
    out_dir = root / "data" / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] 查询 GDC API: {args.project}")
    hits = query_gdc_slides(args.project)
    rows = [flatten(h) for h in hits]
    rows.sort(key=lambda r: r["file_name"])
    print(f"      共 {len(rows)} 张 Slide Image")

    fields = [
        "file_id", "file_name", "file_size", "md5sum", "state", "access",
        "platform", "patient_id", "sample_id", "sample_type", "sample_type_code",
        "disease_type", "primary_site",
    ]

    print("[2/4] 按 sample_type 分组")
    by_type: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_type[row["sample_type"] or "UNKNOWN"].append(row)

    normals = by_type.get("Solid Tissue Normal", [])
    tumors = by_type.get("Primary Tumor", []) + by_type.get("Recurrent Tumor", [])

    print("[3/4] 写出 manifest")
    write_csv(out_dir / "tcga_ucec_all_slides.csv", rows, fields)
    write_csv(out_dir / "tcga_ucec_normals.csv", normals, fields)
    write_csv(out_dir / "tcga_ucec_tumor.csv", tumors, fields)
    write_gdc_manifest(out_dir / "gdc_manifest_normals.txt", normals)
    write_gdc_manifest(out_dir / "gdc_manifest_tumor.txt", tumors)

    patients_normal = {r["patient_id"] for r in normals}
    patients_tumor = {r["patient_id"] for r in tumors}
    common = patients_normal & patients_tumor

    summary = {
        "project": args.project,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_slides": len(rows),
        "total_size_bytes": sum(r["file_size"] for r in rows),
        "total_size_tb": round(sum(r["file_size"] for r in rows) / 1024 ** 4, 3),
        "by_sample_type": {
            k: {
                "slides": len(v),
                "patients": len({r["patient_id"] for r in v}),
                "size_gb": round(sum(r["file_size"] for r in v) / 1024 ** 3, 1),
            }
            for k, v in sorted(by_type.items())
        },
        "sample_type_code_counts": dict(Counter(r["sample_type_code"] for r in rows)),
        "patients_with_normal_and_tumor": len(common),
        "platforms": dict(Counter(r["platform"] for r in rows)),
    }
    (out_dir / "tcga_ucec_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("[4/4] 完成")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
