#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
14_build_split_manifest.py
==========================
从本地可用切片生成监督学习的划分清单（患者级防泄漏）。

为什么必须患者级划分：
    TCGA 同一患者可能贡献多张切片（诊断切片 + 组织切片）。若按切片随机划分，
    同一患者的切片会同时出现在训练与测试集，导致性能虚高。

输出：
    data/manifests/<cohort>_split_manifest.csv
        slide_id, patient_id, sample_type, label, split, cohort, wsi_path, tissue_slide

用法：
    python scripts/14_build_split_manifest.py --cohort tcga_pair --seed 42
    python scripts/14_build_split_manifest.py --cohort tcga_pair --label-mode tissue_normal_vs_tumor
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANI = ROOT / "data" / "manifests"

WSI_EXTS = (".svs", ".tif", ".tiff", ".ndpi")
UUID_RE = re.compile(r"\.[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$")


def canonical_slide_id(name: str) -> str:
    """统一 slide_id：去扩展名 + 去 GDC 文件名的 UUID 段。

    必须与 10_tiling.py / 12_extract_features.py 使用同一规则，否则划分清单与
    特征文件会对不上，导致样本被静默丢弃（已实测：不一致时交集为 0）。
    """
    stem = re.sub(r"\.(svs|h5|tif|tiff|ndpi)$", "", name, flags=re.I)
    return UUID_RE.sub("", stem)


def scan_local() -> list[dict]:
    """扫描本地所有可用切片，关联 GDC 元数据。"""
    master = {}
    master_path = MANI / "slide_master.csv"
    if master_path.exists():
        for r in csv.DictReader(open(master_path, encoding="utf-8-sig")):
            master[r["file_name"]] = r

    rows: list[dict] = []
    for d in [ROOT / "TCGA-UCEC", ROOT / "data" / "raw"]:
        if not d.exists():
            continue
        for ext in WSI_EXTS:
            for p in d.rglob(f"*{ext}"):
                meta = master.get(p.name, {})
                slide_id = canonical_slide_id(p.stem)
                patient = meta.get("patient_id") or "-".join(p.name.split("-")[:3])
                stype = meta.get("sample_type") or ("Primary Tumor" if "-01" in p.name[8:12] else "Unknown")
                code = meta.get("sample_type_code") or (p.name.split("-")[3] if p.name.count("-") > 3 else "")
                rows.append({
                    "slide_id": slide_id,
                    "patient_id": patient,
                    "sample_type": stype,
                    "sample_type_code": code,
                    "wsi_path": str(p.relative_to(ROOT)),
                    "size_bytes": p.stat().st_size,
                })
    # 去重（同一 slide_id 只保留一个路径）
    seen = {}
    for r in rows:
        seen.setdefault(r["slide_id"], r)
    return list(seen.values())


def assign_labels(rows: list[dict], mode: str) -> list[dict]:
    out = []
    for r in rows:
        st = r["sample_type"]
        if mode == "tissue_normal_vs_tumor":
            if st == "Solid Tissue Normal":
                lab = 0
            elif st in ("Primary Tumor", "Recurrent Tumor"):
                lab = 1
            else:
                continue
        elif mode == "diagnostic_only":
            if r["sample_type_code"].startswith("01"):
                lab = 1
            elif r["sample_type_code"].startswith("11"):
                lab = 0
            else:
                continue
        else:
            raise ValueError(f"未知 label-mode: {mode}")
        r = dict(r)
        r["label"] = lab
        out.append(r)
    return out


def patient_split(rows: list[dict], seed: int, ratios=(0.7, 0.15, 0.15)) -> list[dict]:
    by_patient: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_patient[r["patient_id"]].append(r)

    rng = random.Random(seed)
    patients = sorted(by_patient)
    rng.shuffle(patients)
    n = len(patients)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train_p = set(patients[:n_train])
    val_p = set(patients[n_train:n_train + n_val])
    test_p = set(patients[n_train + n_val:])

    out = []
    for p, items in by_patient.items():
        split = "train" if p in train_p else ("val" if p in val_p else "test")
        for r in items:
            r = dict(r)
            r["split"] = split
            out.append(r)
    out.sort(key=lambda r: (r["split"], r["slide_id"]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="tcga_pair")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label-mode", default="tissue_normal_vs_tumor",
                    choices=["tissue_normal_vs_tumor", "diagnostic_only"])
    args = ap.parse_args()

    local = scan_local()
    print(f"本地切片 {len(local)} 张")
    labeled = assign_labels(local, args.label_mode)
    print(f"有标签 {len(labeled)} 张（label-mode={args.label_mode}）")
    if not labeled:
        print("[WARN] 无可用标签切片", file=sys.stderr)
        return 1

    rows = patient_split(labeled, args.seed)

    by_split: dict[str, dict] = defaultdict(lambda: {"n": 0, "pos": 0, "neg": 0, "patients": set()})
    for r in rows:
        s = by_split[r["split"]]
        s["n"] += 1
        s["pos" if r["label"] == 1 else "neg"] += 1
        s["patients"].add(r["patient_id"])

    out_path = MANI / f"{args.cohort}_split_manifest.csv"
    fields = ["slide_id", "patient_id", "sample_type", "sample_type_code",
              "label", "split", "cohort", "wsi_path", "size_bytes"]
    with open(out_path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            r = dict(r)
            r["cohort"] = args.cohort
            w.writerow({k: r.get(k, "") for k in fields})

    print(f"\n划分结果（患者级，seed={args.seed}）：")
    for s in ("train", "val", "test"):
        d = by_split.get(s)
        if not d:
            print(f"  {s:5s} 0 张")
            continue
        print(f"  {s:5s} 切片 {d['n']:3d}  患者 {len(d['patients']):3d}  "
              f"恶性 {d['pos']:3d}  良性/正常 {d['neg']:3d}")

    # 泄漏自检
    train_p = by_split["train"]["patients"]
    test_p = by_split["test"]["patients"]
    overlap = train_p & test_p
    print(f"\n患者级泄漏检查：train∩test = {len(overlap)} 例 "
          f"{'✅ 无泄漏' if not overlap else '❌ 存在泄漏'}")
    print(f"-> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
