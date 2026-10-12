#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
47_probe_zenodo_query.py
========================
定位 Zenodo 400 的具体原因：是 sort 参数、空格、& 还是 H&E 里的 & 符号。

用法：python scripts/47_probe_zenodo_query.py
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

UA = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
BASE = "https://zenodo.org/api/records"


def probe(label: str, url: str) -> bool:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
            d = json.load(r)
            print(f"[OK]   {label}: total={d.get('hits', {}).get('total')}")
            return True
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read()[:160].decode("utf-8", "replace")
        except Exception:  # noqa: BLE001
            pass
        print(f"[HTTP {exc.code}] {label}  {body}")
        return False
    except Exception as exc:  # noqa: BLE001
        print(f"[ERR]  {label}: {type(exc).__name__}")
        return False


def main() -> int:
    q = urllib.parse.quote
    cases = [
        ("single word + sort", f"{BASE}?q=endometrial&size=5&sort=mostrecent"),
        ("single word no sort", f"{BASE}?q=endometrial&size=5"),
        ("two words quoted", f"{BASE}?q={q(chr(34))}endometrial%20cancer{q(chr(34))}&size=5"),
        ("two words unquoted", f"{BASE}?q={q('endometrial cancer')}&size=5"),
        ("two words no sort", f"{BASE}?q={q('endometrial cancer')}&size=5"),
        ("two words + sort", f"{BASE}?q={q('endometrial cancer')}&size=5&sort=mostrecent"),
        ("ampersand in term", f"{BASE}?q={q('endometrium H&E')}&size=5"),
        ("ampersand raw", f"{BASE}?q=endometrium%20H%26E&size=5"),
        ("three words", f"{BASE}?q={q('endometrial whole slide image')}&size=5"),
        ("plus instead of space", f"{BASE}?q=endometrial+cancer&size=5"),
    ]
    ok = sum(1 for label, url in cases if probe(label, url))
    print(f"\n可用 {ok}/{len(cases)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
