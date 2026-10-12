#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
39_fetch_kaggle_zip.py
======================
用「多段并发 + 断点续传 + 大小校验」下载 Kaggle 数据集 zip。

背景：直接顺序下载 Kaggle zip 会在中途被截断（实测 206 MB 只拿到 7.5 MB），
而 pandas/zipfile 都无法从截断文件中恢复。本脚本按 Range 分段下载并校验大小。

用法：
    python scripts/39_fetch_kaggle_zip.py --ref samihamuntahamahin/endometrial-histopathology
    python scripts/39_fetch_kaggle_zip.py --ref ... --segments 8 --workers 8
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UA = {"User-Agent": "Mozilla/5.0 (research)"}


def head_size(url: str) -> int:
    for method in ("HEAD", "GET"):
        try:
            headers = dict(UA)
            if method == "GET":
                headers["Range"] = "bytes=0-0"
            req = urllib.request.Request(url, headers=headers, method=method)
            with urllib.request.urlopen(req, timeout=60) as resp:
                if method == "HEAD":
                    n = resp.headers.get("Content-Length")
                    if n:
                        return int(n)
                else:
                    cr = resp.headers.get("Content-Range") or ""
                    if "/" in cr:
                        return int(cr.split("/")[-1])
        except Exception:  # noqa: BLE001
            continue
    return 0


def fetch_segment(url: str, start: int, end: int, dest: Path, retries: int = 60) -> tuple[int, str]:
    want = end - start + 1
    for attempt in range(1, retries + 1):
        have = dest.stat().st_size if dest.exists() else 0
        if have >= want:
            return have, "ok"
        headers = dict(UA)
        headers["Range"] = f"bytes={start + have}-{end}"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=90) as resp:
                mode = "ab" if have else "wb"
                with open(dest, mode) as fh:
                    while True:
                        chunk = resp.read(512 * 1024)
                        if not chunk:
                            break
                        fh.write(chunk)
        except Exception:  # noqa: BLE001
            time.sleep(min(10, 1 + attempt // 5))
            continue
    have = dest.stat().st_size if dest.exists() else 0
    return have, "ok" if have >= want else "incomplete"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="samihamuntahamahin/endometrial-histopathology")
    ap.add_argument("--segments", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="data/raw/endometrial_patch")
    args = ap.parse_args()

    url = f"https://www.kaggle.com/api/v1/datasets/download/{args.ref}"
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{args.ref.split('/')[-1]}.zip"

    size = head_size(url)
    print(f"数据集 {args.ref}")
    print(f"远端大小：{size/1024**2:.1f} MB" if size else "远端大小：未知（将按段下载后校验）")

    if dest.exists() and size and dest.stat().st_size == size:
        print("已完整下载，跳过")
        return 0
    if dest.exists():
        print(f"已有文件 {dest.stat().st_size/1024**2:.1f} MB，重新下载")
        dest.unlink()

    if not size:
        # 无法确定大小：退化为顺序下载 + 长重试
        part = dest.with_suffix(".zip.part")
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=300) as resp, open(part, "wb") as fh:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
        part.replace(dest)
        print(f"完成 {dest.stat().st_size/1024**2:.1f} MB（未做大小校验）")
        return 0

    n = max(1, min(args.segments, size // (8 * 1024 * 1024)))
    step = size // n
    # 段目录必须**按数据集名区分**：若共用固定目录，多个数据集并行下载时会互相
    # 覆盖同一批 segNNN.part，导致拼装出的文件永远与远端大小不符（已实测）。
    tmp = out_dir / f"._kaggle_segments_{dest.stem}"
    tmp.mkdir(parents=True, exist_ok=True)

    # ⚠️ 段边界依赖 n。若上次用不同 --segments 跑过，目录里会残留**旧边界**的
    # segNNN.part，而拼装循环只按当前 n 逐段顺序读取——多出来的旧文件不会被读到，
    # 缺失的段也不会报错，于是拼出一个大小不符的损坏文件，只在最后打一行 WARN。
    # （已实测：8 段残留 + 12 段新下载 -> 拼出 6812 MB vs 远端 6100 MB。）
    # 因此**拼装前必须校验**：段文件数量与名称完全匹配、且总字节数等于远端大小。
    bounds = [(i * step, (size - 1) if i == n - 1 else (i * step + step - 1)) for i in range(n)]
    print(f"分 {n} 段，段级并发 {args.workers}")

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(fetch_segment, url, s, e, tmp / f"seg{i:03d}.part")
                for i, (s, e) in enumerate(bounds)]
        for i, fut in enumerate(as_completed(futs), 1):
            got, st = fut.result()
            print(f"  段 {i}/{n}: {got/1024**2:.1f} MB {st}", flush=True)

    # ---- 拼装前校验（防止把旧边界的残留段拼进来）
    expected = [tmp / f"seg{i:03d}.part" for i in range(n)]
    missing = [p.name for p in expected if not p.exists()]
    stale = sorted(p.name for p in tmp.glob("seg*.part") if p not in expected)
    if missing or stale:
        print(f"[ERROR] 段文件不匹配，拒绝拼装：缺失 {missing}；多余（可能是旧边界残留）{stale}")
        if stale:
            for p in tmp.glob("seg*.part"):
                if p not in expected:
                    p.unlink()
            print(f"[INFO] 已清理 {len(stale)} 个残留段；请重跑本命令（会重新下载缺失段）")
        return 1
    total = sum(p.stat().st_size for p in expected)
    if total != size:
        print(f"[ERROR] 段总字节 {total} != 远端 {size}，拒绝拼装（请重跑以补齐）")
        return 1

    assembly = out_dir / f".{dest.stem}.assembling"
    with open(assembly, "wb") as out:
        for p in expected:
            with open(p, "rb") as fh:
                while True:
                    c = fh.read(8 * 1024 * 1024)
                    if not c:
                        break
                    out.write(c)
    got = assembly.stat().st_size
    if got != size:
        print(f"[WARN] 大小不符 {got} / {size}")
        return 1
    assembly.replace(dest)
    for p in tmp.glob("*.part"):
        p.unlink()
    tmp.rmdir()
    secs = time.time() - t0
    print(f"\n完成 {dest.stat().st_size/1024**2:.1f} MB，用时 {secs/60:.1f} min，"
          f"平均 {got/1024**2/max(secs,1):.2f} MB/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
