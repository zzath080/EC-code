#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
28_build_01z_manifest.py
========================
生成「仅 01Z 诊断切片」的下载清单。

为什么需要它：
    gdc_manifest_tumor.txt 含**全部**肿瘤切片（01A/01B 组织切片 + 01Z 诊断切片）。
    按体量升序下载会优先拿到小的 01A 组织切片（约 80 MB），而不是跨标本类型实验
    所需的 01Z 诊断切片（约 1.5 GB）。本脚本按条码第 4 段精确筛选。

同时输出分层抽样清单：按体量分位数分层，避免只拿到最小的一批切片导致样本偏倚。

输出：
    data/manifests/gdc_manifest_01z.txt            全部 01Z
    data/manifests/gdc_manifest_01z_stratified.txt 分层抽样（默认 120 张）
    data/manifests/01z_summary.json                统计
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANI = ROOT / "data" / "manifests"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-size", type=int, default=120, help="分层抽样张数")
    ap.add_argument("--code", default="01Z", help="要筛选的条码第 4 段")
    args = ap.parse_args()

    src = MANI / "tcga_ucec_all_slides.csv"
    rows = list(csv.DictReader(open(src, encoding="utf-8-sig")))

    def fsize(r: dict) -> int:
        # 兼容不同来源的字段名（00_build_gdc_manifest.py 输出的是 file_size）
        for key in ("file_size", "size_bytes", "size"):
            if r.get(key) not in (None, ""):
                return int(r[key])
        return 0

    picked = [r for r in rows if (r.get("sample_type_code") or "").startswith(args.code)]
    # 排除已在本地的（按文件名匹配略繁琐，这里不做排除，下载器会自动 SKIP）

    picked.sort(key=fsize)
    total = sum(fsize(r) for r in picked)

    def write_manifest(path: Path, items: list[dict]) -> None:
        with open(path, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh, delimiter="\t")
            w.writerow(["id", "filename", "md5", "size", "state"])
            for r in items:
                w.writerow([r["file_id"], r["file_name"], r["md5sum"], fsize(r), "released"])

    write_manifest(MANI / f"gdc_manifest_{args.code.lower()}.txt", picked)

    # 按体量四分层抽样，保证大小分布不被体量偏斜
    n = len(picked)
    k = min(args.sample_size, n)
    strata = 4
    per = k // strata
    sampled: list[dict] = []
    for s in range(strata):
        lo = s * n // strata
        hi = (s + 1) * n // strata
        chunk = picked[lo:hi]
        if not chunk:
            continue
        step = max(1, len(chunk) // per) if per else 1
        sampled.extend(chunk[::step][:per])
    # 去重并补齐
    seen = set()
    uniq = []
    for r in sampled:
        if r["file_id"] not in seen:
            seen.add(r["file_id"])
            uniq.append(r)
    if len(uniq) < k:
        for r in picked:
            if len(uniq) >= k:
                break
            if r["file_id"] not in seen:
                seen.add(r["file_id"])
                uniq.append(r)
    uniq.sort(key=fsize)
    write_manifest(MANI / f"gdc_manifest_{args.code.lower()}_stratified.txt", uniq)

    sizes = sorted(fsize(r) for r in picked)
    usizes = sorted(fsize(r) for r in uniq)
    summary = {
        "code": args.code,
        "n_total": n,
        "total_gb": round(total / 1024 ** 3, 1),
        "median_mb": round(sizes[n // 2] / 1024 ** 2, 1) if n else 0,
        "min_mb": round(sizes[0] / 1024 ** 2, 1) if n else 0,
        "max_mb": round(sizes[-1] / 1024 ** 2, 1) if n else 0,
        "n_stratified": len(uniq),
        "stratified_gb": round(sum(usizes) / 1024 ** 3, 2),
        "n_patients_total": len({r["patient_id"] for r in picked}),
        "n_patients_stratified": len({r["patient_id"] for r in uniq}),
    }
    (MANI / f"{args.code.lower()}_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n-> {MANI / f'gdc_manifest_{args.code.lower()}.txt'}")
    print(f"-> {MANI / f'gdc_manifest_{args.code.lower()}_stratified.txt'}")
    if summary["max_mb"] > 1024:
        print(f"\n提示：最大切片 {summary['max_mb']:.0f} MB，单张较大；"
              f"若磁盘紧张可减少 --sample-size")
    return 0


if __name__ == "__main__":
    sys.exit(main())
