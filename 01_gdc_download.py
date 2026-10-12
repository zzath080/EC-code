#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
01_gdc_download.py
==================
纯 Python 的 GDC 切片下载器：多线程分块、断点续传、MD5 校验。

为什么不直接用 gdc-client：
    - gdc-client 不支持按 sample_type / 大小 / 随机子集``挑选文件；
    - 无法在下载过程中写入自定义命名（研究里需要 slide_id 稳定的目录结构）；
    - 不支持"先下小文件、按需增量"的预算控制策略。
    本脚本保留 gdc-client 的 manifest 格式兼容，可混用。

用法：
    # 下载 54 张正常组织（负类），并发 6
    python scripts/01_gdc_download.py --manifest data/manifests/gdc_manifest_normals.txt \
        --out data/raw/tcga_ucec_normals --workers 6

    # 按预算下载肿瘤切片：先小后大，最多 200 GB
    python scripts/01_gdc_download.py --manifest data/manifests/gdc_manifest_tumor.txt \
        --out data/raw/tcga_ucec_tumor --budget-gb 200 --order size_asc

    # 只列出将要下载的内容，不实际下载
    python scripts/01_gdc_download.py --manifest ... --dry-run

    # 校验已下载文件的 MD5
    python scripts/01_gdc_download.py --manifest ... --out ... --verify-only

输出：
    <out>/<slide_id>.svs              切片文件
    <out>/_download_state.json        每个文件的完成状态（断点续传依据）
    <out>/_download_log.csv           下载日志（大小、耗时、速率、校验结果）
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

GDC_DATA_ENDPOINT = "https://api.gdc.cancer.gov/data/{file_id}"
CHUNK = 1024 * 1024  # 1 MiB
# 本机网络对大文件会反复掐断连接（实测每 15–30 MB 断一次），
# 因此重试次数必须给足，否则接近完成的下载会被反复丢弃。
RETRIES = 60

_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def read_manifest(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if not row.get("id"):
                continue
            rows.append({
                "file_id": row["id"].strip(),
                "file_name": row["filename"].strip(),
                "md5": (row.get("md5") or "").strip().lower(),
                "size": int(row.get("size") or 0),
                "state": (row.get("state") or "").strip(),
            })
    return rows


def slide_id_from_name(file_name: str) -> str:
    """TCGA-BG-A0MG-01Z-00-DX1.<UUID>.svs -> TCGA-BG-A0MG-01Z-00-DX1"""
    stem = file_name
    for ext in (".svs", ".tif", ".tiff", ".ndpi", ".mrxs"):
        if stem.lower().endswith(ext):
            stem = stem[: -len(ext)]
            break
    return stem


def md5_of(path: Path, block: int = 8 * 1024 * 1024) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            data = fh.read(block)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def load_state(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_state(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def select_rows(rows: list[dict], order: str, limit: int | None, budget_bytes: int | None) -> list[dict]:
    rows = list(rows)
    if order == "size_asc":
        rows.sort(key=lambda r: r["size"])
    elif order == "size_desc":
        rows.sort(key=lambda r: -r["size"])
    elif order == "random":
        random.Random(42).shuffle(rows)
    else:  # name
        rows.sort(key=lambda r: r["file_name"])

    if limit is not None:
        rows = rows[:limit]
    if budget_bytes:
        picked, total = [], 0
        for row in rows:
            if total + row["size"] > budget_bytes:
                continue
            picked.append(row)
            total += row["size"]
        rows = picked
    return rows


def download_one(row: dict, out_dir: Path, state: dict, state_path: Path, state_lock: threading.Lock) -> dict:
    file_id = row["file_id"]
    slide_id = slide_id_from_name(row["file_name"])
    dest = out_dir / f"{slide_id}.svs"
    part = out_dir / f".{slide_id}.svs.part"
    start = time.time()

    if dest.exists() and row["size"] and dest.stat().st_size == row["size"]:
        digest = md5_of(dest) if row["md5"] else ""
        ok = (not row["md5"]) or digest == row["md5"]
        with state_lock:
            state[file_id] = {"slide_id": slide_id, "status": "done" if ok else "md5_mismatch",
                              "size": dest.stat().st_size, "md5": digest}
            save_state(state_path, state)
        log(f"  [SKIP] {slide_id} 已完成{'（校验通过）' if ok else '（MD5 不符）'}")
        return {"slide_id": slide_id, "status": "skip" if ok else "md5_mismatch",
                "bytes": dest.stat().st_size, "seconds": 0.0}

    for attempt in range(1, RETRIES + 1):
        try:
            existing = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={existing}-"} if existing else {}
            req = urllib.request.Request(GDC_DATA_ENDPOINT.format(file_id=file_id), headers=headers)
            with urllib.request.urlopen(req, timeout=180) as resp:
                # GDC 在限流时可能忽略 Range 直接返回 200（全量）。此时若继续以
                # "ab" 追加会拼出损坏文件，因此必须按响应实际语义决定写入模式：
                #   206 -> 服务端接受续传，用 "ab"
                #   200 + 已有部分 -> 服务端从头给，用 "wb" 覆盖重来
                resume_ok = (existing > 0 and resp.status == 206)
                mode = "ab" if resume_ok else "wb"
                if mode == "wb":
                    existing = 0
                with open(part, mode) as fh:
                    while True:
                        chunk = resp.read(CHUNK)
                        if not chunk:
                            break
                        fh.write(chunk)
            got = part.stat().st_size
            if row["size"] and got != row["size"]:
                raise IOError(f"size mismatch got={got} want={row['size']}")
            if row["md5"] and md5_of(part) != row["md5"]:
                part.unlink(missing_ok=True)
                raise IOError("md5 mismatch, discarded partial file")
            os.replace(part, dest)
            secs = time.time() - start
            mbps = got / 1024 ** 2 / secs if secs > 0 else 0
            with state_lock:
                state[file_id] = {"slide_id": slide_id, "status": "done", "size": got,
                                  "md5": row["md5"], "seconds": round(secs, 1)}
                save_state(state_path, state)
            log(f"  [OK] {slide_id}  {got/1024**3:.2f} GB  {secs:.0f}s  {mbps:.1f} MB/s")
            return {"slide_id": slide_id, "status": "ok", "bytes": got, "seconds": secs, "mbps": mbps}
        except Exception as exc:  # noqa: BLE001
            now = part.stat().st_size if part.exists() else 0
            if attempt % 5 == 1 or attempt == RETRIES:
                log(f"  [WARN] {slide_id} 第 {attempt}/{RETRIES} 次失败"
                    f"（已续传 {now/1024**2:.0f} MB）: {type(exc).__name__}")
            # GDC 限流时退避更久，避免持续触发服务端保护
            time.sleep(min(20, 2 + attempt))

    with state_lock:
        state[file_id] = {"slide_id": slide_id, "status": "failed", "size": 0}
        save_state(state_path, state)
    return {"slide_id": slide_id, "status": "failed", "bytes": 0, "seconds": time.time() - start}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", type=Path, help="输出目录")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="最多下载 N 张")
    ap.add_argument("--budget-gb", type=float, default=None, help="下载总量预算（GB）")
    ap.add_argument("--order", choices=["name", "size_asc", "size_desc", "random"], default="size_asc")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    rows = read_manifest(args.manifest)
    rows = [r for r in rows if r["state"] in ("released", "submitted", "")]
    total_all = sum(r["size"] for r in rows)
    print(f"manifest: {args.manifest}  条目 {len(rows)}  合计 {total_all/1024**3:.1f} GB")

    rows = select_rows(rows, args.order, args.limit, int(args.budget_gb * 1024**3) if args.budget_gb else None)
    total = sum(r["size"] for r in rows)
    print(f"本次选中 {len(rows)} 张，合计 {total/1024**3:.1f} GB，顺序={args.order}")

    if args.dry_run:
        for r in rows[:20]:
            print(f"  {slide_id_from_name(r['file_name'])}  {r['size']/1024**2:.0f} MB")
        if len(rows) > 20:
            print(f"  ... 其余 {len(rows)-20} 张")
        return 0

    if not args.out:
        print("[ERROR] 需要 --out", file=sys.stderr)
        return 2
    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "_download_state.json"
    state = load_state(state_path)
    state_lock = threading.Lock()

    if args.verify_only:
        bad = 0
        for row in rows:
            sid = slide_id_from_name(row["file_name"])
            f = out_dir / f"{sid}.svs"
            if not f.exists():
                print(f"  [MISSING] {sid}")
                bad += 1
                continue
            if row["md5"] and md5_of(f) != row["md5"]:
                print(f"  [BAD-MD5] {sid}")
                bad += 1
        print(f"校验完成，问题文件 {bad}/{len(rows)}")
        return 0

    started = time.time()
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, r, out_dir, state, state_path, state_lock): r for r in rows}
        for fut in as_completed(futures):
            results.append(fut.result())

    ok = [r for r in results if r["status"] in ("ok", "skip")]
    failed = [r for r in results if r["status"] not in ("ok", "skip")]
    secs = time.time() - started
    got = sum(r["bytes"] for r in ok)

    with open(out_dir / "_download_log.csv", "a", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        if fh.tell() == 0:
            w.writerow(["slide_id", "status", "bytes", "seconds", "mbps", "finished_at"])
        for r in results:
            w.writerow([r["slide_id"], r["status"], r.get("bytes", 0), round(r.get("seconds", 0), 1),
                        round(r.get("mbps", 0), 2), time.strftime("%Y-%m-%dT%H:%M:%S")])

    print(f"\n完成：成功 {len(ok)}  失败 {len(failed)}  共 {got/1024**3:.2f} GB  用时 {secs/60:.1f} min")
    if failed:
        print("失败清单：")
        for r in failed:
            print(f"  {r['slide_id']}")
    (out_dir / "_download_summary.json").write_text(json.dumps({
        "manifest": str(args.manifest), "selected": len(rows), "ok": len(ok), "failed": len(failed),
        "bytes": got, "seconds": round(secs, 1), "workers": args.workers,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
