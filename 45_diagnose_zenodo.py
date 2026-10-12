#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
45_diagnose_zenodo.py
=====================
诊断 Zenodo API 检索为何返回 400，并找出可用的查询写法。

用法：python scripts/45_diagnose_zenodo.py
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (endo-wsi-research)", "Accept": "application/json"}

BASE = "https://zenodo.org/api/records"


def probe(label: str, url: str) -> dict:
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=60) as resp:
            d = json.load(resp)
            total = d.get("hits", {}).get("total")
            print(f"[OK]   {label}: total={total}")
            return {"ok": True, "total": total, "url": url}
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read()[:200].decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            pass
        print(f"[HTTP {exc.code}] {label}: {body}")
        return {"ok": False, "status": exc.code, "body": body, "url": url}
    except Exception as exc:  # noqa: BLE001
        print(f"[ERR]  {label}: {type(exc).__name__}: {exc}")
        return {"ok": False, "error": str(exc), "url": url}


def main() -> int:
    q = urllib.parse.quote
    cases = [
        ("bare term", f"{BASE}?q=endometrial&size=5"),
        ("quoted term", f"{BASE}?q={q(chr(34) + 'endometrial' + chr(34))}&size=5"),
        ("with type", f"{BASE}?q=endometrial&type=dataset&size=5"),
        ("with sort", f"{BASE}?q=endometrial&sort=mostrecent&size=5"),
        ("trailing slash", f"{BASE}/?q=endometrial&size=5"),
        ("records/ path", "https://zenodo.org/api/records?q=endometrial&size=5&page=1"),
        ("simple no size", f"{BASE}?q=endometrial"),
        ("multi-word", f"{BASE}?q={q('endometrium histopathology')}&size=5"),
        ("search api", "https://zenodo.org/api/records?q=endometrial"),
    ]
    results = []
    for label, url in cases:
        results.append({"label": label, **probe(label, url)})

    ok = [r for r in results if r.get("ok")]
    print(f"\n可用写法 {len(ok)}/{len(results)}")
    if ok:
        print("建议使用：", ok[0]["url"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
