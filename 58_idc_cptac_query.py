#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
58_idc_cptac_query.py
=====================
用 idc-index 查询 CPTAC-UCEC 的病理切片（SM）明细，评估下载与读取可行性。

要回答的问题：
    1. CPTAC-UCEC 到底有多少张病理 WSI、总体量多大？
    2. 都是 DICOM 吗？有没有可直接用 OpenSlide 读的 SVS/TIFF？
    3. 是否包含"正常/良性"样本（决定能否做良恶性任务）？

用法：
    python scripts/58_idc_cptac_query.py
    python scripts/58_idc_cptac_query.py --download-sm --limit 3   # 试下一例
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / ".pylibs"
if LIB.exists():
    sys.path.insert(0, str(LIB))
OUT = ROOT / "data" / "manifests" / "idc_cptac"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download-sm", action="store_true")
    ap.add_argument("--limit", type=int, default=3)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    try:
        import idc_index
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] 无法导入 idc_index：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    print(f"idc_index 版本：{getattr(idc_index, '__version__', '?')}")
    client = None
    for attr in ("IDCClient", "IdcClient"):
        if hasattr(idc_index, attr):
            client = getattr(idc_index, attr)()
            print(f"已实例化 {attr}")
            break
    if client is None:
        print("[ERROR] 未找到 IDCClient 类；可用的顶层属性：",
              [a for a in dir(idc_index) if not a.startswith("_")][:40])
        return 2

    # 查询 CPTAC-UCEC 的 SM（slide microscopy）文件
    print("\n查询 CPTAC-UCEC 的 Slide Microscopy 文件 …")
    try:
        df = client.sql_query(
            "SELECT collection_id, Modality, SeriesInstanceUID, SOPInstanceUID, "
            "instance_size, PatientID, SeriesDescription "
            "FROM index WHERE collection_id = 'cptac_ucec' AND Modality = 'SM'"
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] SQL 查询失败：{type(exc).__name__}: {exc}")
        return 2

    print(f"返回行数：{len(df)}")
    if len(df) == 0:
        print("[WARN] 没有 SM 文件")
        return 1
    print(f"列：{list(df.columns)}")
    print(df.head(5).to_string())

    # 汇总
    summary: dict = {"n_rows": int(len(df))}
    if "instance_size" in df.columns:
        total = float(df["instance_size"].astype(float).sum())
        summary["total_gb"] = round(total / 1024 ** 3, 2)
    if "PatientID" in df.columns:
        summary["n_patients"] = int(df["PatientID"].nunique())
        summary["n_series"] = int(df["SeriesInstanceUID"].nunique()) if "SeriesInstanceUID" in df.columns else None
    if "SeriesDescription" in df.columns:
        summary["series_description_top"] = dict(Counter(df["SeriesDescription"].astype(str)).most_common(15))

    print("\n=== 汇总 ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    (OUT / "cptac_ucec_sm_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # 每例切片数分布
    if "PatientID" in df.columns:
        per = df.groupby("PatientID").size()
        print(f"\n每例 DICOM 实例数：min={per.min()} 中位={int(per.median())} max={per.max()}")

    if args.download_sm:
        print(f"\n=== 试下载 1 例（前 {args.limit} 个 series）===")
        try:
            series = list(df["SeriesInstanceUID"].unique())[: args.limit]
            client.download_from_selection(series_instance_uid=series,
                                           download_dir=str(ROOT / "data" / "raw" / "cptac_ucec"))
            print("下载完成 -> data/raw/cptac_ucec")
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] 下载失败：{type(exc).__name__}: {exc}")

    print("\n=== 读取可行性 ===")
    print("- IDC 的病理切片为 DICOM（SM），**OpenSlide 不能直接读**；")
    print("- 可选读取方案：pydicom（取像素）+ 自建金字塔，或 dcmtk/dicom2tiff 转换，")
    print("  或使用 `highdicom`；转换后再走现有分块流程。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
