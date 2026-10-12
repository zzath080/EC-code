#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
34_parallel_download.py
=======================
多段并发下载器（针对 GDC 这类支持 Range 的服务端）。

实测依据（scripts/33_parallel_range_probe.py）：
    单连接        0.709 MB/s
    8 段并发聚合  3.574 MB/s   → 5.0× 提升
因此本脚本对**每个文件**内部做多段并发，而不是靠"多文件并行"（后者会被服务端限流）。

特性：
    - 每文件切成 ~N 段，各自独立续传；段级 .part 文件保留，可随时中断恢复
    - 全部段完成后按序拼接 → 校验大小与 MD5 → 原子重命名
    - 段级失败只重试该段，不丢弃已完成部分
    - 支持预算/抽样/断点续传

用法：
    python scripts/34_parallel_download.py --manifest data/manifests/gdc_manifest_01z_stratified.txt \
        --out data/raw/tcga_ucec_01z --segments 8 --workers 8
    python scripts/34_parallel_download.py --manifest ... --out ... --files-parallel 2
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

GDC = "https://api.gdc.cancer.gov/data/{fid}"
SEG_MIN_BYTES = 16 * 1024 * 1024       # 段最小 16 MB，避免过多小段
UA = {"User-Agent": "gdc-client/2.3"}

_lock = threading.Lock()


def log(msg: str) -> None:
    with _lock:
        print(msg, flush=True)


def read_manifest(p: Path) -> list[dict]:
    rows = []
    for r in csv.DictReader(open(p, encoding="utf-8-sig"), delimiter="\t"):
        if not r.get("id"):
            continue
        rows.append({"file_id": r["id"].strip(), "filename": r["filename"].strip(),
                     "md5": (r.get("md5") or "").strip().lower(),
                     "size": int(r.get("size") or 0)})
    return rows


def slide_id(name: str) -> str:
    for ext in (".svs", ".tif", ".tiff", ".ndpi"):
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return name


def md5_file(p: Path, block: int = 8 * 1024 * 1024) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        while True:
            d = fh.read(block)
            if not d:
                break
            h.update(d)
    return h.hexdigest()


def fetch_segment(fid: str, start: int, end: int, dest: Path, retries: int = 40) -> tuple[int, str]:
    """下载 [start, end] 闭区间到 dest，支持续传。返回 (字节数, 状态)。"""
    want = end - start + 1
    for attempt in range(1, retries + 1):
        have = dest.stat().st_size if dest.exists() else 0
        if have >= want:
            return have, "ok"
        headers = dict(UA)
        headers["Range"] = f"bytes={start + have}-{end}"
        try:
            req = urllib.request.Request(GDC.format(fid=fid), headers=headers)
            with urllib.request.urlopen(req, timeout=90) as resp:
                if resp.status not in (200, 206):
                    raise IOError(f"unexpected status {resp.status}")
                mode = "ab" if have else "wb"
                with open(dest, mode) as fh:
                    while True:
                        chunk = resp.read(512 * 1024)
                        if not chunk:
                            break
                        fh.write(chunk)
        except Exception:  # noqa: BLE001
            time.sleep(min(15, 1 + attempt // 5))
            continue
    have = dest.stat().st_size if dest.exists() else 0
    return have, "ok" if have >= want else "incomplete"


def download_file(row: dict, out_dir: Path, segments: int, seg_workers: int) -> dict:
    sid = slide_id(row["filename"])
    dest = out_dir / f"{sid}.svs"
    tmp_dir = out_dir / "._segments" / sid
    size = row["size"]

    if dest.exists() and size and dest.stat().st_size == size:
        return {"slide_id": sid, "status": "skip", "bytes": size, "seconds": 0.0}

    tmp_dir.mkdir(parents=True, exist_ok=True)
    n = max(1, min(segments, max(1, size // SEG_MIN_BYTES)))
    bounds = []
    step = size // n
    for i in range(n):
        s = i * step
        e = (size - 1) if i == n - 1 else (s + step - 1)
        bounds.append((s, e))

    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=seg_workers) as pool:
        futs = {pool.submit(fetch_segment, row["file_id"], s, e, tmp_dir / f"seg{i:03d}.part"): i
                for i, (s, e) in enumerate(bounds)}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                results.append((i, *fut.result()))
            except Exception as exc:  # noqa: BLE001
                results.append((i, 0, f"error:{type(exc).__name__}"))

    bad = [r for r in results if r[2] != "ok"]
    if bad:
        return {"slide_id": sid, "status": "failed", "bytes": 0,
                "seconds": time.time() - t0, "detail": f"{len(bad)}/{n} 段未完成"}

    # 按序拼接
    assembly = out_dir / f".{sid}.assembling"
    with open(assembly, "wb") as out:
        for i in range(n):
            seg_path = tmp_dir / f"seg{i:03d}.part"
            with open(seg_path, "rb") as fh:
                while True:
                    chunk = fh.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
    got = assembly.stat().st_size
    if size and got != size:
        return {"slide_id": sid, "status": "size_mismatch", "bytes": got, "seconds": time.time() - t0}
    if row["md5"] and md5_file(assembly) != row["md5"]:
        assembly.unlink(missing_ok=True)
        return {"slide_id": sid, "status": "md5_mismatch", "bytes": got, "seconds": time.time() - t0}

    os.replace(assembly, dest)
    for p in tmp_dir.glob("*.part"):
        p.unlink(missing_ok=True)
    tmp_dir.rmdir()
    secs = time.time() - t0
    return {"slide_id": sid, "status": "ok", "bytes": got, "seconds": secs,
            "mbps": round(got / 1024 ** 2 / secs, 2) if secs > 0 else 0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--segments", type=int, default=8, help="每个文件切成几段")
    ap.add_argument("--workers", type=int, default=8, help="段级并发数")
    ap.add_argument("--files-parallel", type=int, default=1, help="同时处理几个文件")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--budget-gb", type=float, default=None)
    ap.add_argument("--order", choices=["name", "size_asc", "size_desc"], default="size_asc")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = read_manifest(Path(args.manifest))
    if args.order == "size_asc":
        rows.sort(key=lambda r: r["size"])
    elif args.order == "size_desc":
        rows.sort(key=lambda r: -r["size"])
    if args.limit:
        rows = rows[: args.limit]
    if args.budget_gb:
        picked, tot = [], 0
        for r in rows:
            if tot + r["size"] > args.budget_gb * 1024 ** 3:
                continue
            picked.append(r)
            tot += r["size"]
        rows = picked

    total = sum(r["size"] for r in rows)
    print(f"清单 {args.manifest}：{len(rows)} 个文件，{total/1024**3:.2f} GB")
    print(f"策略：每文件 {args.segments} 段，段级并发 {args.workers}，文件级并发 {args.files_parallel}")
    if args.dry_run:
        for r in rows[:10]:
            print(f"  {slide_id(r['filename'])[:52]:54s} {r['size']/1024**2:8.1f} MB")
        return 0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    done, failed = [], []
    if args.files_parallel <= 1:
        for i, row in enumerate(rows, 1):
            res = download_file(row, out_dir, args.segments, args.workers)
            (done if res["status"] in ("ok", "skip") else failed).append(res)
            log(f"[{i}/{len(rows)}] {res['status']:14s} {res['slide_id'][:44]:46s} "
                f"{res.get('bytes',0)/1024**3:.2f} GB {res.get('seconds',0):6.0f}s "
                f"{res.get('mbps','')} MB/s {res.get('detail','')}")
    else:
        with ThreadPoolExecutor(max_workers=args.files_parallel) as pool:
            futs = {pool.submit(download_file, r, out_dir, args.segments, args.workers): r for r in rows}
            for i, fut in enumerate(as_completed(futs), 1):
                res = fut.result()
                (done if res["status"] in ("ok", "skip") else failed).append(res)
                log(f"[{i}/{len(rows)}] {res['status']:14s} {res['slide_id'][:44]:46s} "
                    f"{res.get('bytes',0)/1024**3:.2f} GB {res.get('seconds',0):6.0f}s "
                    f"{res.get('mbps','')} MB/s")

    got = sum(r.get("bytes", 0) for r in done)
    secs = time.time() - t0
    print(f"\n完成：成功 {len(done)}  失败 {len(failed)}  共 {got/1024**3:.2f} GB  "
          f"用时 {secs/60:.1f} min  平均 {got/1024**2/max(secs,1):.2f} MB/s")
    for r in failed:
        print(f"  失败：{r['slide_id']} {r.get('detail','')}")
    (out_dir / "_parallel_download_summary.json").write_text(json.dumps({
        "manifest": args.manifest, "segments": args.segments, "workers": args.workers,
        "n_ok": len(done), "n_failed": len(failed), "bytes": got,
        "seconds": round(secs, 1), "avg_mbps": round(got / 1024 ** 2 / max(secs, 1), 3),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
