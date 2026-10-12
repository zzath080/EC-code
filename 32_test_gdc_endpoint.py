#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
32_test_gdc_endpoint.py
=======================
诊断 GDC 数据端点连通性：区分「服务端限流/中断」与「本机网络问题」。

用法：
    python scripts/32_test_gdc_endpoint.py
    python scripts/32_test_gdc_endpoint.py --manifest data/manifests/gdc_manifest_01z_stratified.txt --n 5
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def probe(file_id: str, nbytes: int = 4096) -> dict:
    url = f"https://api.gdc.cancer.gov/data/{file_id}"
    rec: dict = {"url": url}
    t0 = time.time()
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "gdc-client/2.3",
            "Range": f"bytes=0-{nbytes - 1}",
        })
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read(nbytes)
            rec.update({"status": resp.status, "got": len(body), "seconds": round(time.time() - t0, 2),
                        "content_length": resp.headers.get("Content-Length"),
                        "content_range": resp.headers.get("Content-Range"),
                        "ok": len(body) > 0})
    except urllib.error.HTTPError as exc:
        rec.update({"status": exc.code, "error": str(exc.reason), "ok": False})
    except Exception as exc:  # noqa: BLE001
        rec.update({"status": "ERR", "error": f"{type(exc).__name__}: {exc}", "ok": False})
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifests/gdc_manifest_01z_stratified.txt")
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()

    p = ROOT / args.manifest
    rows = list(csv.DictReader(open(p, encoding="utf-8"), delimiter="\t"))
    print(f"清单 {p.name}：{len(rows)} 条，测试前 {args.n} 条\n")

    ok = 0
    for r in rows[: args.n]:
        rec = probe(r["id"])
        flag = "OK  " if rec.get("ok") else "FAIL"
        print(f"[{flag}] {r['filename'][:52]:54s} status={rec.get('status')} "
              f"got={rec.get('got', 0)}B {rec.get('seconds', '')}s {rec.get('error', '')}")
        ok += 1 if rec.get("ok") else 0
        time.sleep(0.5)

    print(f"\n连通 {ok}/{args.n}")
    if ok == 0:
        print("→ GDC 端点当前对本机不可用（可能是服务端限流或本地网络）；"
              "可稍后重试，或改用 gdc-client 官方客户端。")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
