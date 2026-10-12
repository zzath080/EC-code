#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
33_parallel_range_probe.py
==========================
测试 GDC 是否支持「多连接并行分段下载」——本机单连接吞吐只有 0.1–0.8 MB/s，
若服务端支持并发 range，则可以成倍提速。

做法：对同一文件发起 N 个并发的 Range 请求，测量各段实际吞吐与总吞吐。

用法：
    python scripts/33_parallel_range_probe.py --workers 8 --mb-per-seg 4
"""

from __future__ import annotations

import argparse
import csv
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GDC = "https://api.gdc.cancer.gov/data/{fid}"


def timed_range(fid: str, start: int, length: int, out: dict, key: str) -> None:
    url = GDC.format(fid=fid)
    hdr = {"User-Agent": "gdc-client/2.3", "Range": f"bytes={start}-{start + length - 1}"}
    t0 = time.time()
    try:
        req = urllib.request.Request(url, headers=hdr)
        with urllib.request.urlopen(req, timeout=120) as resp:
            got = 0
            while got < length:
                chunk = resp.read(min(256 * 1024, length - got))
                if not chunk:
                    break
                got += len(chunk)
            dt = time.time() - t0
            out[key] = {"bytes": got, "seconds": round(dt, 2),
                        "mbps": round(got / 1024 ** 2 / dt, 3) if dt > 0 else 0,
                        "status": resp.status}
    except Exception as exc:  # noqa: BLE001
        out[key] = {"bytes": 0, "seconds": round(time.time() - t0, 2), "mbps": 0,
                    "status": f"{type(exc).__name__}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifests/gdc_manifest_01z_stratified.txt")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--mb-per-seg", type=float, default=4.0)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(ROOT / args.manifest, encoding="utf-8"), delimiter="\t"))
    # 选一个较大的文件，确保 range 有意义
    rows.sort(key=lambda r: -int(r["size"]))
    target = rows[3] if len(rows) > 3 else rows[0]
    fid = target["id"]
    total_mb = int(target["size"]) / 1024 ** 2
    seg = int(args.mb_per_seg * 1024 ** 2)

    print(f"目标：{target['filename'][:56]}")
    print(f"总大小 {total_mb:.0f} MB，并发 {args.workers} 段 × {args.mb_per_seg:.0f} MB\n")

    # 先测单连接吞吐作为基线
    base: dict = {}
    timed_range(fid, 0, seg, base, "single")
    s = base["single"]
    print(f"单连接：{s['bytes']/1024**2:.1f} MB / {s['seconds']}s = {s['mbps']} MB/s  (status {s['status']})")

    # 并发分段（各段取文件中不同位置，避免同一段被缓存影响）
    out: dict = {}
    threads = []
    t0 = time.time()
    for i in range(args.workers):
        start = int(total_mb * 1024 ** 2) // (args.workers + 1) * (i + 1)
        th = threading.Thread(target=timed_range, args=(fid, start, seg, out, f"seg{i}"))
        threads.append(th)
        th.start()
    for th in threads:
        th.join()
    wall = time.time() - t0

    ok = [v for v in out.values() if isinstance(v.get("status"), int) and v["bytes"] > 0]
    got = sum(v["bytes"] for v in ok)
    print(f"\n并发 {args.workers} 段：成功 {len(ok)}/{args.workers}")
    for k, v in sorted(out.items()):
        print(f"  {k}: {v['bytes']/1024**2:6.1f} MB  {v['seconds']:6.2f}s  {v['mbps']:6.3f} MB/s  status={v['status']}")
    agg = got / 1024 ** 2 / wall if wall > 0 else 0
    print(f"\n并发聚合吞吐：{agg:.3f} MB/s（墙钟 {wall:.1f}s，共 {got/1024**2:.1f} MB）")
    print(f"相对单连接提升：{agg / max(s['mbps'], 1e-9):.1f}×")

    if agg > s["mbps"] * 1.5:
        print("\n→ 结论：服务端**支持并行分段**，应把下载器改为多段并发，可显著提速。")
    else:
        print("\n→ 结论：并发无显著提升，瓶颈在总带宽或服务端限流；增加连接数无用。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
