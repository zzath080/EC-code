#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
46_search_zenodo_wsi.py
=======================
在 Zenodo 上深度检索子宫内膜 / 子宫组织病理的 **WSI 或图像数据集**，
并把候选按「是否可能含良恶性标签」「体量」「许可」筛出来。

用法：
    python scripts/46_search_zenodo_wsi.py
    python scripts/46_search_zenodo_wsi.py --download-best --max-gb 20
输出：
    data/manifests/zenodo_wsi_candidates.json
    data/manifests/zenodo_wsi_candidates.md
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "manifests"
UA = {"User-Agent": "Mozilla/5.0 (endo-wsi-research)", "Accept": "application/json"}
BASE = "https://zenodo.org/api/records"

QUERIES = [
    "endometrial whole slide image",
    "endometrium histopathology dataset",
    "uterus histology whole slide",
    "endometrial cancer histopathology",
    "endometrial biopsy image",
    "uterine pathology dataset",
    "endometrium H&E image dataset",
]

# 与良恶性判别相关的关键词（用于打分）
MALIGNANT_KW = ["malignan", "cancer", "carcinoma", "adenocarcinoma", "tumor", "tumour", "neoplas"]
BENIGN_KW = ["benign", "normal", "hyperplasia", "polyp", "atrophic", "proliferative", "secretory"]
IMAGE_KW = ["image", "wsi", "whole slide", "histolog", "histopatholog", "patch", "h&e", "he-stain", "tile"]


def search(q: str, size: int = 40) -> list[dict]:
    url = f"{BASE}?q={urllib.parse.quote(q)}&size={size}&sort=mostrecent"
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=90) as resp:
        d = json.load(resp)
    out = []
    for h in d.get("hits", {}).get("hits", []):
        md = h.get("metadata", {}) or {}
        files = h.get("files", []) or []
        title = str(md.get("title") or "")
        desc = str(md.get("description") or "")
        blob = (title + " " + desc).lower()
        lic = md.get("license")
        if isinstance(lic, dict):
            lic = lic.get("id")
        out.append({
            "id": h.get("id"),
            "doi": h.get("doi"),
            "title": title[:160],
            "type": (md.get("resource_type") or {}).get("type"),
            "license": lic,
            "access": h.get("access_right"),
            "n_files": len(files),
            "total_bytes": sum(f.get("size", 0) for f in files),
            "file_names": [f.get("key") for f in files][:8],
            "has_image_kw": any(k in blob for k in IMAGE_KW),
            "has_malignant_kw": any(k in blob for k in MALIGNANT_KW),
            "has_benign_kw": any(k in blob for k in BENIGN_KW),
            "matched_query": q,
            "url": f"https://zenodo.org/records/{h.get('id')}",
        })
    return out


def score(r: dict) -> int:
    s = 0
    if r["has_image_kw"]:
        s += 2
    if r["has_malignant_kw"]:
        s += 2
    if r["has_benign_kw"]:
        s += 2          # 同时含良恶性关键词最关键
    if r["access"] == "open":
        s += 2
    if r["total_bytes"] > 1e9:
        s += 1
    if (r["license"] or "").lower().startswith(("cc0", "cc-by")):
        s += 1
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download-best", action="store_true")
    ap.add_argument("--max-gb", type=float, default=20.0)
    args = ap.parse_args()

    seen: dict[str, dict] = {}
    per_query: dict[str, int] = {}
    for q in QUERIES:
        try:
            rows = search(q)
        except Exception as exc:  # noqa: BLE001
            print(f"[{q}] 失败：{type(exc).__name__}: {exc}")
            continue
        per_query[q] = len(rows)
        for r in rows:
            key = str(r["id"])
            if key in seen:
                seen[key]["matched_query"] += " | " + q
            else:
                seen[key] = r

    cands = list(seen.values())
    # 过滤：必须是图像相关
    cands = [c for c in cands if c["has_image_kw"]]
    for c in cands:
        c["score"] = score(c)
    cands.sort(key=lambda c: (-c["score"], -c["total_bytes"]))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "zenodo_wsi_candidates.json").write_text(
        json.dumps({"per_query": per_query, "candidates": cands}, indent=2, ensure_ascii=False),
        encoding="utf-8")

    lines = ["# Zenodo 上子宫内膜/子宫组织病理图像数据集候选", "",
             f"检索词：{len(QUERIES)} 个，命中图像相关候选 **{len(cands)}** 个", "",
             "| 分数 | 标题 | 体量 | 许可 | 良/恶关键词 | DOI |", "| --- | --- | --- | --- | --- | --- |"]
    for c in cands[:30]:
        gb = c["total_bytes"] / 1024 ** 3
        kw = ("恶" if c["has_malignant_kw"] else "-") + ("/良" if c["has_benign_kw"] else "")
        lines.append(f"| {c['score']} | {c['title'][:70]} | {gb:.2f} GB | {c['license']} | {kw} | {c['doi']} |")
    (OUT / "zenodo_wsi_candidates.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\n图像相关候选 {len(cands)} 个；前 15：")
    for c in cands[:15]:
        print(f"  [{c['score']}] {c['title'][:62]:64s} {c['total_bytes']/1024**3:7.2f} GB  "
              f"{str(c['license'])[:12]:14s} 恶={c['has_malignant_kw']} 良={c['has_benign_kw']}")
    print(f"\n-> {OUT/'zenodo_wsi_candidates.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
