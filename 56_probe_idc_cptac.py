#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
56_probe_idc_cptac.py
=====================
探查 IDC（NCI Imaging Data Commons）上 CPTAC-UCEC 的**病理全切片**能否实际下载。

背景（已核实）：
    - CPTAC-UCEC 是**独立于 TCGA 的子宫内膜癌队列**，IDC 收录 254 例，
      模态含 SM（Slide Microscopy，即病理 WSI）
    - 元数据经 Zenodo record 13351631 发布（CC BY 3.0），manifest 仅 ~60 KB，
      实际文件托管在 IDC 公开的 AWS/GCS 桶上
    - 下载方式：`pip install idc-index` 后 `idc download manifest.s5cmd`

价值：这是唯一能立刻拿到的、独立于 TCGA 的子宫内膜癌 WSI 队列，
可用于「跨数据来源」泛化验证（主设计的第 4 个队列）。

本脚本：
    1. 下载 Zenodo 上的三个 manifest（很小）
    2. 解析 manifest 头部说明与文件规模
    3. 判断是否需要 idc-index，并给出可执行的下一步

用法：
    python scripts/56_probe_idc_cptac.py
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "manifests" / "idc_cptac"
UA = {"User-Agent": "Mozilla/5.0 (endo-wsi-research)"}
RECORD = "https://zenodo.org/api/records/13351631"


def get_json(url: str) -> dict:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90) as r:
        return json.load(r)


def fetch_text(url: str) -> str:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
        return r.read().decode("utf-8", "replace")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("查询 Zenodo record 13351631（CPTAC-UCEC DICOM SM）…")
    rec = get_json(RECORD)
    files = rec.get("files", [])
    print(f"manifest 文件 {len(files)} 个：")
    for f in files:
        print(f"  {f['key']}  {f['size']/1024:.1f} KB")

    dl = {}
    for f in files:
        key = f["key"]
        try:
            text = fetch_text(f["links"]["self"])
            (OUT / key).write_text(text, encoding="utf-8")
            dl[key] = text
            print(f"  已下载 {key}（{len(text)} 字符）")
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL] {key}: {type(exc).__name__}: {exc}")

    s5 = next((k for k in dl if k.endswith(".s5cmd")), None)
    if s5:
        text = dl[s5]
        lines = [l for l in text.splitlines() if l.strip()]
        header = [l for l in lines if l.startswith("#")][:12]
        data_lines = [l for l in lines if not l.startswith("#")]
        print(f"\n--- {s5} 头部说明 ---")
        for h in header:
            print("  " + h[:160])
        print(f"\n数据行 {len(data_lines)} 条；样例：")
        for d in data_lines[:5]:
            print("  " + d[:170])

        # 统计扩展名与目录，估算规模
        exts = Counter()
        for d in data_lines:
            for tok in d.split():
                if "." in tok:
                    exts[tok.rsplit(".", 1)[-1].lower()] += 1
        print(f"\n扩展名分布（前 10）：{dict(exts.most_common(10))}")

    print("\n=== 下一步 ===")
    print("需要 idc-index 才能按 manifest 下载（它会处理 AWS/GCS 认证与路径）。")
    print("  python scripts/pipx.py --target .pylibs install idc-index")
    print("  python -m idc download <manifest>.s5cmd")
    print("\n注意：病理切片为 DICOM（SM）格式，读取需要 dcmtk/pydicom/highdicom 之一；")
    print("      OpenSlide 不能直接读 DICOM WSI，因此后续分块流程需要另做适配。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
