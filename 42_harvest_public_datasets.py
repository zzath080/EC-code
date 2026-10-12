#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
42_harvest_public_datasets.py
=============================
系统化检索并获取公开的子宫内膜/子宫组织病理数据源。

背景：项目范围限定为**可公开直接获取**的数据，因此需要系统检索
**无需专有格式、且含良恶性标签**的公开数据，构成可执行的备选研究设计。

检索平台（均用官方 API，不做网页抓取）：
    - HuggingFace datasets  : https://huggingface.co/api/datasets
    - Kaggle datasets        : https://www.kaggle.com/api/v1/datasets/list
    - Zenodo                 : https://zenodo.org/api/records
    - Figshare               : https://api.figshare.com/v2/articles/search
    - OpenAlex（论文→数据）   : https://api.openalex.org/works

用法：
    python scripts/42_harvest_public_datasets.py --search
    python scripts/42_harvest_public_datasets.py --search --download-kaggle
输出：
    data/manifests/public_dataset_harvest.json
    data/manifests/public_dataset_harvest.md
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "manifests"
UA = {"User-Agent": "Mozilla/5.0 (endo-wsi-research)"}

TERMS = ["endometrial", "endometrium", "uterus histopathology", "uterine pathology",
         "endometrial cancer histology", "endometrial biopsy"]


def get_json(url: str, timeout: int = 60):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def safe(name: str, fn):
    try:
        return fn()
    except urllib.error.HTTPError as exc:
        print(f"    [{name}] HTTP {exc.code}")
    except Exception as exc:  # noqa: BLE001
        print(f"    [{name}] {type(exc).__name__}: {exc}")
    return []


def hf_datasets(term: str) -> list[dict]:
    d = get_json(f"https://huggingface.co/api/datasets?search={urllib.parse.quote(term)}&limit=50")
    return [{"id": x.get("id"), "gated": x.get("gated"), "downloads": x.get("downloads"),
             "likes": x.get("likes"), "lastModified": x.get("lastModified")} for x in d]


def kaggle_datasets(term: str) -> list[dict]:
    d = get_json(f"https://www.kaggle.com/api/v1/datasets/list?search={urllib.parse.quote(term)}")
    out = []
    for x in d if isinstance(d, list) else []:
        if not isinstance(x, dict):
            continue
        out.append({"ref": x.get("ref"), "title": x.get("titleNullable") or x.get("title"),
                    "subtitle": x.get("subtitleNullable") or x.get("subtitle"),
                    "bytes": x.get("totalBytesNullable") or x.get("totalBytes"),
                    "license": x.get("licenseNameNullable") or x.get("licenseName"),
                    "downloads": x.get("downloadCount"), "votes": x.get("voteCount")})
    return out


def zenodo_records(term: str) -> list[dict]:
    d = get_json(f"https://zenodo.org/api/records?q={urllib.parse.quote(term)}&size=30")
    out = []
    for h in d.get("hits", {}).get("hits", []):
        md = h.get("metadata", {})
        files = h.get("files", [])
        out.append({"id": h.get("id"), "doi": h.get("doi"), "title": md.get("title"),
                    "type": (md.get("resource_type") or {}).get("type"),
                    "license": (md.get("license") or {}).get("id") if isinstance(md.get("license"), dict) else md.get("license"),
                    "n_files": len(files),
                    "total_bytes": sum(f.get("size", 0) for f in files),
                    "access": h.get("access_right")})
    return out


def figshare_search(term: str) -> list[dict]:
    payload = json.dumps({"search_for": term, "page_size": 30}).encode()
    req = urllib.request.Request("https://api.figshare.com/v2/articles/search",
                                 data=payload, headers={**UA, "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        d = json.load(resp)
    return [{"id": x.get("id"), "title": x.get("title"), "doi": x.get("doi"),
             "published": x.get("published_date"), "url": x.get("url_public_html")} for x in d]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--download-kaggle", action="store_true")
    ap.add_argument("--min-bytes", type=int, default=5_000_000, help="Kaggle 候选最小体量")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    harvest: dict = {"terms": TERMS, "huggingface": {}, "kaggle": {}, "zenodo": {}, "figshare": {}}

    print("=== HuggingFace ===")
    for t in TERMS:
        r = safe(f"hf:{t}", lambda t=t: hf_datasets(t))
        if r:
            harvest["huggingface"][t] = r
            for x in r:
                print(f"  [{t}] {x['id']}  gated={x['gated']}  dl={x['downloads']}")

    print("\n=== Kaggle ===")
    for t in TERMS:
        r = safe(f"kaggle:{t}", lambda t=t: kaggle_datasets(t))
        if r:
            harvest["kaggle"][t] = r
            for x in r[:8]:
                mb = (x["bytes"] or 0) / 1024 ** 2
                print(f"  [{t}] {x['ref']}  {mb:.1f} MB  {x['license']}  {str(x['subtitle'])[:60]}")

    print("\n=== Zenodo ===")
    for t in ["endometrial", "endometrium histopathology"]:
        r = safe(f"zenodo:{t}", lambda t=t: zenodo_records(t))
        if r:
            harvest["zenodo"][t] = r
            for x in r[:8]:
                print(f"  [{t}] {str(x['title'])[:60]}  {(x['total_bytes'] or 0)/1024**3:.2f} GB  {x['license']}")

    print("\n=== Figshare ===")
    for t in ["endometrial histopathology", "endometrium dataset"]:
        r = safe(f"figshare:{t}", lambda t=t: figshare_search(t))
        if r:
            harvest["figshare"][t] = r
            for x in r[:8]:
                print(f"  [{t}] {str(x['title'])[:60]}  {x['doi']}")

    (OUT / "public_dataset_harvest.json").write_text(
        json.dumps(harvest, indent=2, ensure_ascii=False), encoding="utf-8")

    # 汇总候选：Kaggle 上体量足够、许可宽松的
    cands: dict[str, dict] = {}
    for t, rows in harvest["kaggle"].items():
        for x in rows:
            ref = x.get("ref")
            if not ref:
                continue
            b = x.get("bytes") or 0
            if b >= args.min_bytes:
                cands.setdefault(ref, {**x, "matched_terms": []})
                cands[ref]["matched_terms"].append(t)
    lines = ["# 公开子宫内膜数据源检索汇总", "",
             f"检索词：{', '.join(TERMS)}", "",
             "## Kaggle 候选（体量 ≥ %.0f MB）" % (args.min_bytes / 1024 ** 2), "",
             "| ref | 标题 | 体量(MB) | 许可 | 下载数 |", "| --- | --- | --- | --- | --- |"]
    for ref, x in sorted(cands.items(), key=lambda kv: -(kv[1]["bytes"] or 0)):
        lines.append(f"| {ref} | {str(x['title'])[:50]} | {(x['bytes'] or 0)/1024**2:.1f} | "
                     f"{x['license']} | {x['downloads']} |")
    lines += ["", "## HuggingFace 命中", "", "| id | gated | downloads |", "| --- | --- | --- |"]
    for t, rows in harvest["huggingface"].items():
        for x in rows:
            lines.append(f"| {x['id']} | {x['gated']} | {x['downloads']} |")
    (OUT / "public_dataset_harvest.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\nKaggle 候选 {len(cands)} 个")
    print(f"-> {OUT/'public_dataset_harvest.json'}")
    print(f"-> {OUT/'public_dataset_harvest.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
