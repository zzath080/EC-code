#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
49_download_external_wsi.py
===========================
下载独立的外部验证 WSI 数据（ENDO-AID 真实活检队列）。

目标：Endometrium Carcinoma Pipelle biopsies（Radboud / ENDO-AID）
     Zenodo record 7372187，91 张 Pipelle 活检 WSI，TIFF@0.5µm/px，CC BY-NC 4.0。
     单个 `endometrium-carcinoma-pipelle.rar` 约 16.9 GB。

用途定位（重要）：
     该数据集**全部为恶性**，**不能**做良恶性训练；它的价值是
     (a) 提供一个**独立于 TCGA** 的子宫内膜活检域，用于检验模型在外部数据上的
         假阳性与校准表现；
     (b) 是唯一可直接获取的**活检** WSI 队列。

特性：分段并发下载 + 段级续传 + 大小校验；Zenodo 会限流，因此带指数退避。

用法：
    python scripts/49_download_external_wsi.py --probe        # 只测连通性
    python scripts/49_download_external_wsi.py --download
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
UA = {"User-Agent": "Mozilla/5.0 (endo-wsi-research)"}
RECORD_API = "https://zenodo.org/api/records/7372187"
OUT = ROOT / "data" / "raw" / "endoaid_external"


def get_json(url: str, retries: int = 6) -> dict:
    last = None
    for i in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90) as r:
                return json.load(r)
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2 ** i)
    raise RuntimeError(f"无法获取 {url}: {last}")


def probe(url: str, label: str) -> dict:
    try:
        req = urllib.request.Request(url, headers={**UA, "Range": "bytes=0-1023"})
        with urllib.request.urlopen(req, timeout=90) as resp:
            b = resp.read(1024)
            rec = {"label": label, "status": resp.status, "bytes": len(b),
                   "content_range": resp.headers.get("Content-Range"),
                   "head": b[:8].decode("latin-1")}
            print(f"[OK] {label}: status={rec['status']} bytes={rec['bytes']} "
                  f"range={rec['content_range']} head={rec['head']!r}")
            return rec
    except urllib.error.HTTPError as exc:
        print(f"[HTTP {exc.code}] {label}")
        return {"label": label, "status": exc.code, "ok": False}
    except Exception as exc:  # noqa: BLE001
        print(f"[ERR] {label}: {type(exc).__name__}: {exc}")
        return {"label": label, "error": str(exc), "ok": False}


def fetch_segment(url: str, start: int, end: int, dest: Path, retries: int = 300) -> tuple[int, str]:
    """下载一段，支持续传。

    Zenodo 对并发连接限流较激进，实测 8 段各约 1.2 GB 后会集中出现 URLError，
    因此重试上限必须给足（段文件本身可续传，重试成本低）。
    """
    want = end - start + 1
    last = ""
    for attempt in range(1, retries + 1):
        have = dest.stat().st_size if dest.exists() else 0
        if have >= want:
            return have, "ok"
        headers = {**UA, "Range": f"bytes={start + have}-{end}"}
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as resp:
                mode = "ab" if have else "wb"
                with open(dest, mode) as fh:
                    while True:
                        chunk = resp.read(512 * 1024)
                        if not chunk:
                            break
                        fh.write(chunk)
        except Exception as exc:  # noqa: BLE001
            last = type(exc).__name__
            # 每 20 次打印一次进度，便于观察是否仍在前进
            if attempt % 20 == 1:
                now = dest.stat().st_size if dest.exists() else 0
                print(f"    seg({start}) 续传中 第 {attempt} 次 {now/1024**2:.0f} MB: {last}", flush=True)
            # Zenodo 限流较激进，退避到最多 30 秒（过长会拖慢整体）
            time.sleep(min(30, 1 + attempt // 4))
            continue
    have = dest.stat().st_size if dest.exists() else 0
    return have, "ok" if have >= want else f"incomplete:{last}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--segments", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    print("查询 Zenodo 记录 7372187 …")
    rec = get_json(RECORD_API)
    files = rec.get("files", [])
    for f in files:
        print(f"  {f['key']}  {f['size']/1024**3:.2f} GB  md5={f.get('checksum')}")
    target = next((f for f in files if f["key"].endswith(".rar")), None)
    if not target:
        print("[ERROR] 未找到 .rar 文件", file=sys.stderr)
        return 2

    url = target["links"]["self"]
    size = int(target["size"])

    if args.probe or not args.download:
        probe(url, target["key"])
        return 0

    dest = OUT / target["key"]
    if dest.exists() and dest.stat().st_size == size:
        print(f"已完整下载：{dest}")
        return 0

    n = max(1, min(args.segments, size // (64 * 1024 * 1024)))
    step = size // n
    tmp = OUT / "._segments"
    tmp.mkdir(parents=True, exist_ok=True)
    print(f"\n分 {n} 段并发下载，段级并发 {args.workers}，目标 {size/1024**3:.2f} GB")

    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {}
        for i in range(n):
            s = i * step
            e = (size - 1) if i == n - 1 else (s + step - 1)
            futs[pool.submit(fetch_segment, url, s, e, tmp / f"seg{i:03d}.part")] = i
        for fut in as_completed(futs):
            i = futs[fut]
            got, st = fut.result()
            results.append((i, got, st))
            print(f"  段 {i}: {got/1024**2:8.1f} MB  {st}", flush=True)

    bad = [r for r in results if r[2] != "ok"]
    if bad:
        print(f"\n[WARN] {len(bad)}/{n} 段未完成，保留 .part 以便下次续传")
        (OUT / "download_state.json").write_text(json.dumps(
            {"segments_done": n - len(bad), "segments_total": n, "bytes": size}, indent=2), encoding="utf-8")
        return 1

    assembly = OUT / f".{target['key']}.assembling"
    with open(assembly, "wb") as out:
        for i in range(n):
            with open(tmp / f"seg{i:03d}.part", "rb") as fh:
                while True:
                    c = fh.read(16 * 1024 * 1024)
                    if not c:
                        break
                    out.write(c)
    got = assembly.stat().st_size
    if got != size:
        print(f"[WARN] 拼接后大小不符 {got} / {size}")
        return 1
    assembly.replace(dest)
    for p in tmp.glob("*.part"):
        p.unlink()
    tmp.rmdir()
    secs = time.time() - t0
    print(f"\n完成 {dest.name}  {got/1024**3:.2f} GB  {secs/60:.1f} min  "
          f"平均 {got/1024**2/max(secs,1):.2f} MB/s")
    print("注意：.rar 需要解压；本机若无 unrar，可用 7-Zip 或 Python 的 rarfile（需 unrar 后端）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
