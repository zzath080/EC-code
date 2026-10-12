#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
48_find_datasets_via_literature.py
==================================
当数据平台（Zenodo/Figshare）检索被限流时，改从**论文元数据**反查数据集：
用 Crossref / Europe PMC 搜子宫内膜组织病理相关论文，提取其摘要与元数据中的
数据集线索（Zenodo/Figshare/IEEE DataPort/GitHub 链接、DOI），再逐个核实。

用法：
    python scripts/48_find_datasets_via_literature.py
输出：
    data/manifests/literature_dataset_leads.json
    data/manifests/literature_dataset_leads.md
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "manifests"
UA = {"User-Agent": "Mozilla/5.0 (endo-wsi-research)"}

CROSSREF = "https://api.crossref.org/works"
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

QUERIES = [
    "endometrial whole slide image dataset",
    "endometrium histopathology dataset deep learning",
    "endometrial cancer whole slide image classification",
    "uterine histopathology dataset machine learning",
    "endometrial biopsy artificial intelligence dataset",
]

LINK_RE = re.compile(
    r"(https?://(?:www\.)?(?:zenodo\.org|doi\.org/10\.5281/zenodo|figshare\.com|"
    r"data\.mendeley\.com|ieee-dataport\.org|github\.com|huggingface\.co|kaggle\.com)/[^\s\)\]\}\"'<>]+)",
    re.I)


def crossref_search(q: str, rows: int = 20) -> list[dict]:
    url = f"{CROSSREF}?query={urllib.parse.quote(q)}&rows={rows}&select=DOI,title,abstract,URL,issued,container-title"
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=90) as resp:
        d = json.load(resp)
    out = []
    for it in d.get("message", {}).get("items", []):
        title = (it.get("title") or [""])[0]
        abstract = it.get("abstract") or ""
        url_ = it.get("URL") or ""
        blob = " ".join([title, abstract, url_])
        out.append({
            "doi": it.get("DOI"),
            "title": title[:200],
            "journal": (it.get("container-title") or [""])[0],
            "year": (it.get("issued", {}).get("date-parts") or [[None]])[0][0],
            "abstract_snippet": re.sub(r"<[^>]+>", " ", abstract)[:400],
            "links": sorted(set(LINK_RE.findall(blob)))[:10],
        })
    return out


def epmc_search(q: str, page_size: int = 20) -> list[dict]:
    url = f"{EPMC}?query={urllib.parse.quote(q)}&format=json&pageSize={page_size}&resultType=core"
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=90) as resp:
        d = json.load(resp)
    out = []
    for r in d.get("resultList", {}).get("result", []):
        abstract = r.get("abstractText") or ""
        blob = " ".join([str(r.get("title") or ""), abstract])
        out.append({
            "pmid": r.get("pmid"), "doi": r.get("doi"),
            "title": str(r.get("title") or "")[:200],
            "year": r.get("pubYear"),
            "journal": (r.get("journalInfo") or {}).get("journal", {}).get("title"),
            "is_oa": r.get("isOpenAccess"),
            "abstract_snippet": re.sub(r"<[^>]+>", " ", abstract)[:400],
            "links": sorted(set(LINK_RE.findall(blob)))[:10],
        })
    return out


def main() -> int:
    leads: dict = {"crossref": {}, "epmc": {}}
    for q in QUERIES:
        try:
            leads["crossref"][q] = crossref_search(q)
            print(f"[crossref] {q}: {len(leads['crossref'][q])}")
        except Exception as exc:  # noqa: BLE001
            print(f"[crossref] {q} 失败 {type(exc).__name__}: {exc}")
        time.sleep(0.5)
        try:
            leads["epmc"][q] = epmc_search(q)
            print(f"[epmc]     {q}: {len(leads['epmc'][q])}")
        except Exception as exc:  # noqa: BLE001
            print(f"[epmc]     {q} 失败 {type(exc).__name__}: {exc}")
        time.sleep(0.5)

    # 汇总所有数据集链接
    link_hits: dict[str, list[str]] = {}
    paper_titles: dict[str, str] = {}
    for src in ("crossref", "epmc"):
        for q, rows in leads[src].items():
            for r in rows:
                for l in r.get("links", []):
                    clean = l.rstrip(".,;")
                    link_hits.setdefault(clean, [])
                    tag = f"{r.get('doi') or r.get('pmid')}"
                    if tag not in link_hits[clean]:
                        link_hits[clean].append(tag)
                    paper_titles[clean] = r.get("title", "")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "literature_dataset_leads.json").write_text(
        json.dumps({"leads": leads, "link_hits": link_hits}, indent=2, ensure_ascii=False),
        encoding="utf-8")

    lines = ["# 从论文反查数据集链接", "",
             f"检索 {len(QUERIES)} 个查询，命中数据集/代码链接 **{len(link_hits)}** 条", "",
             "| 链接 | 关联论文 DOI/PMID |", "| --- | --- |"]
    for l, papers in sorted(link_hits.items()):
        lines.append(f"| {l[:110]} | {', '.join(papers[:3])} |")
    (OUT / "literature_dataset_leads.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\n命中链接 {len(link_hits)} 条；前 15：")
    for l, papers in list(sorted(link_hits.items()))[:15]:
        print(f"  {l[:100]}  <- {papers[0] if papers else ''}")
    print(f"\n-> {OUT/'literature_dataset_leads.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
