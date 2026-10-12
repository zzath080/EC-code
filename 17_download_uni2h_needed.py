#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
17_download_uni2h_needed.py
===========================
只下载「本地已有对应 WSI」的 UNI2-h 特征文件（由 16_match_uni2h_to_local.py 产出清单）。

这样可以用极小的流量验证 HDF5 读取链路，避免为用不上的特征下载 57 GB。

用法：
    python scripts/17_download_uni2h_needed.py --dry-run
    python scripts/17_download_uni2h_needed.py
    python scripts/17_download_uni2h_needed.py --max-gb 5
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANI = ROOT / "data" / "manifests" / "uni2h"
sys.path.insert(0, str(ROOT / "scripts"))

from pipx import download  # noqa: E402  复用带读超时的下载器


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-gb", type=float, default=None)
    args = ap.parse_args()

    src = MANI / "needed_features.csv"
    if not src.exists():
        print("[ERROR] 先跑 scripts/16_match_uni2h_to_local.py", file=sys.stderr)
        return 2
    rows = list(csv.DictReader(open(src, encoding="utf-8-sig")))
    rows.sort(key=lambda r: int(r["size"]))

    cache = ROOT / "wheels" / "_hf_cache"
    dest_dir = ROOT / "data" / "features" / "uni2h_public" / "TCGA-UCEC"

    total = 0
    picked = []
    for r in rows:
        n = int(r["size"])
        if args.max_gb and total + n > args.max_gb * 1024 ** 3:
            continue
        picked.append(r)
        total += n

    print(f"需要下载 {len(picked)} 个特征文件，合计 {total/1024**3:.2f} GB -> {dest_dir}")
    for r in picked:
        print(f"  {r['slide_id']:40s} {int(r['size'])/1024**2:8.1f} MB  ({r['sample_type']})")
    if args.dry_run:
        return 0

    ok, fail = 0, 0
    for i, r in enumerate(picked, 1):
        try:
            cached = download(r["url"], cache)
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / Path(r["url"]).name
            if not dest.exists() or dest.stat().st_size != cached.stat().st_size:
                import shutil
                shutil.copyfile(cached, dest)
            print(f"[{i}/{len(picked)}] OK   {dest.name}  {dest.stat().st_size/1024**2:.1f} MB", flush=True)
            ok += 1
        except Exception as exc:  # noqa: BLE001
            print(f"[{i}/{len(picked)}] FAIL {r['slide_id']}: {type(exc).__name__}: {exc}", flush=True)
            fail += 1

    print(f"\n完成：成功 {ok}，失败 {fail}；特征目录 {dest_dir}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
