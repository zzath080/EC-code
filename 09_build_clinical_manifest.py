#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
09_build_clinical_manifest.py
=============================
从 GDC 拉取 TCGA-UCEC 的临床与病理元数据，构建论文 Table 1（队列特征表）的数据来源。

获取字段：
    - 人口学：age_at_index, gender, race, ethnicity, vital_status, days_to_death
    - 诊断：primary_diagnosis, morphology, tumor_grade, tumor_stage, figo_stage,
            histological_type, ajcc_pathologic_stage
    - 治疗：prior_treatment, treatment_type
    - 样本：sample_type, tissue_source_site

输出：
    data/manifests/tcga_ucec_clinical.csv       每例一行
    data/manifests/tcga_ucec_clinical_summary.json  汇总统计（可直接写进 Table 1）
"""

from __future__ import annotations

import csv
import json
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANI = ROOT / "data" / "manifests"
ENDPOINT = "https://api.gdc.cancer.gov/cases"

FIELDS = [
    "submitter_id",
    "demographic.age_at_index",
    "demographic.gender",
    "demographic.race",
    "demographic.ethnicity",
    "demographic.vital_status",
    "demographic.days_to_death",
    "diagnoses.primary_diagnosis",
    "diagnoses.morphology",
    "diagnoses.tumor_grade",
    "diagnoses.tumor_stage",
    "diagnoses.figo_stage",
    "diagnoses.ajcc_pathologic_stage",
    "diagnoses.age_at_diagnosis",
    "diagnoses.prior_treatment",
    "diagnoses.tumor_grade",
    "samples.sample_type",
    "samples.submitter_id",
    "project.project_id",
]


def fetch_cases(project: str, page: int = 500) -> list[dict]:
    out: list[dict] = []
    frm = 0
    while True:
        payload = {
            "filters": {"op": "in", "content": {"field": "project.project_id", "value": [project]}},
            "fields": ",".join(FIELDS),
            "format": "JSON",
            "size": page,
            "from": frm,
        }
        req = urllib.request.Request(ENDPOINT, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.load(resp)["data"]
        batch = data["hits"]
        out.extend(batch)
        total = data["pagination"]["total"]
        print(f"  cases {len(out)}/{total}", flush=True)
        frm += page
        if len(out) >= total or not batch:
            break
        time.sleep(0.3)
    return out


def flatten(case: dict) -> dict:
    demo = (case.get("demographic") or {})
    dx = (case.get("diagnoses") or [{}])[0] if case.get("diagnoses") else {}
    samples = case.get("samples") or []
    sample_types = sorted({s.get("sample_type", "") for s in samples})
    return {
        "case_id": case.get("submitter_id", ""),
        "age_at_index": demo.get("age_at_index", ""),
        "gender": demo.get("gender", ""),
        "race": demo.get("race", ""),
        "ethnicity": demo.get("ethnicity", ""),
        "vital_status": demo.get("vital_status", ""),
        "days_to_death": demo.get("days_to_death", ""),
        "primary_diagnosis": dx.get("primary_diagnosis", ""),
        "morphology": dx.get("morphology", ""),
        "tumor_grade": dx.get("tumor_grade", ""),
        "tumor_stage": dx.get("tumor_stage", ""),
        "figo_stage": dx.get("figo_stage", ""),
        "ajcc_pathologic_stage": dx.get("ajcc_pathologic_stage", ""),
        "prior_treatment": dx.get("prior_treatment", ""),
        "sample_types": ";".join(sample_types),
        "n_samples": len(samples),
    }


def main() -> int:
    MANI.mkdir(parents=True, exist_ok=True)
    project = sys.argv[1] if len(sys.argv) > 1 else "TCGA-UCEC"
    print(f"拉取 {project} 临床元数据 …")
    cases = fetch_cases(project)
    rows = [flatten(c) for c in cases]
    rows.sort(key=lambda r: r["case_id"])

    fields = list(rows[0].keys()) if rows else []
    with open(MANI / "tcga_ucec_clinical.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    ages = [float(r["age_at_index"]) for r in rows if r["age_at_index"] not in ("", None)]
    # 样本标准差（ddof=1）。此前只算 mean/min/max，导致 Table 1 的年龄格只能写成
    # "63.9 ± —"（破折号占位）——这是统计字段缺失，不是数据缺失。
    _mean = sum(ages) / len(ages) if ages else None
    _sd = (sum((a - _mean) ** 2 for a in ages) / (len(ages) - 1)) ** 0.5 if len(ages) > 1 else None
    summary = {
        "project": project,
        "n_cases": len(rows),
        "primary_diagnosis": dict(Counter(r["primary_diagnosis"] for r in rows).most_common()),
        "histology_counts": dict(Counter(r["morphology"] for r in rows).most_common(12)),
        "tumor_grade": dict(Counter(r["tumor_grade"] or "NA" for r in rows).most_common()),
        "figo_stage": dict(Counter(r["figo_stage"] or "NA" for r in rows).most_common(12)),
        "ajcc_stage": dict(Counter(r["ajcc_pathologic_stage"] or "NA" for r in rows).most_common(12)),
        "sample_types": dict(Counter(r["sample_types"] for r in rows).most_common(10)),
        "vital_status": dict(Counter(r["vital_status"] or "NA" for r in rows)),
        "age_stats": {
            "n": len(ages),
            "mean": round(_mean, 1) if _mean is not None else None,
            "sd": round(_sd, 1) if _sd is not None else None,
            "min": min(ages) if ages else None,
            "max": max(ages) if ages else None,
        },
    }
    (MANI / "tcga_ucec_clinical_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False)[:2500])
    print(f"\n-> {MANI/'tcga_ucec_clinical.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
