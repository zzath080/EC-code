#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
50_mine_dataset_mentions.py
===========================
更精细地从文献中挖掘数据集线索：不只看摘要，而是抓取开放获取论文的
**数据可用性声明（Data Availability Statement）** 与全文，从中提取数据集链接。

这比单纯检索摘要有效得多，因为数据集链接几乎总在 Data Availability 里。

做法：
    1. Europe PMC 搜 OA 论文（OPEN_ACCESS:Y）并限定含 WSI/组织病理关键词；
    2. 对每篇取 fullTextXML，抽取「data availability / data availability statement」段落；
    3. 从这些段落里正则提取所有数据仓库链接。

用法：
    python scripts/50_mine_dataset_mentions.py --top 40
输出：
    data/manifests/dataset_mentions_from_papers.json
    data/manifests/dataset_mentions_from_papers.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "manifests"
UA = {"User-Agent": "Mozilla/5.0 (endo-wsi-research)"}
EPMC_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EPMC_FULLTEXT = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"

REPOS = ("zenodo.org", "figshare.com", "datadryad.org", "mendeley.com",
         "ieee-dataport.org", "huggingface.co", "kaggle.com", "osf.io",
         "github.com", "biostudies", "ebi.ac.uk", "cancerimagingarchive.net",
         "gdc.cancer.gov", "grand-challenge.org", "syncproject", "aws", "gcs")

LINK_RE = re.compile(r"https?://[^\s\)\]\}\"'<>]+", re.I)

QUERY = ('(endometrial OR endometrium OR uterus) AND '
         '("whole slide" OR histopatholog* OR histolog*) AND '
         '(dataset OR "data availability" OR cohort) AND OPEN_ACCESS:Y')


def epmc_search(query: str, page_size: int) -> list[dict]:
    url = f"{EPMC_SEARCH}?query={urllib.parse.quote(query)}&format=json&pageSize={page_size}&resultType=core"
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90) as resp:
        d = json.load(resp)
    return d.get("resultList", {}).get("result", [])


def fetch_fulltext(pmcid: str) -> str | None:
    try:
        with urllib.request.urlopen(urllib.request.Request(
                EPMC_FULLTEXT.format(pmcid=pmcid), headers=UA), timeout=90) as resp:
            return resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return None if exc.code in (404, 406) else None
    except Exception:  # noqa: BLE001
        return None


def extract_availability(xml: str) -> str:
    """抽取数据可用性相关段落（去掉 XML 标签）。"""
    text = re.sub(r"<[^>]+>", " ", xml)
    text = re.sub(r"\s+", " ", text)
    chunks = []
    for kw in ("data availability", "availability of data", "data and code availability",
               "data accessibility", "accession", "data are available", "dataset is available"):
        for m in re.finditer(kw, text, re.I):
            chunks.append(text[max(0, m.start() - 200): m.start() + 1500])
    return " ".join(chunks)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=40, help="最多处理多少篇 OA 论文")
    args = ap.parse_args()

    print("Europe PMC 检索 OA 论文 …")
    try:
        rows = epmc_search(QUERY, args.top)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] 检索失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(f"命中 {len(rows)} 篇")

    results = []
    links: dict[str, list[str]] = {}
    for i, r in enumerate(rows, 1):
        pmcid = r.get("pmcid")
        rec = {"pmid": r.get("pmid"), "pmcid": pmcid, "doi": r.get("doi"),
               "title": str(r.get("title") or "")[:180], "year": r.get("pubYear"),
               "journal": (r.get("journalInfo") or {}).get("journal", {}).get("title"),
               "links": [], "availability_snippet": ""}
        if pmcid:
            xml = fetch_fulltext(pmcid)
            if xml:
                avail = extract_availability(xml)
                rec["availability_snippet"] = avail[:600]
                found = set()
                for l in LINK_RE.findall(avail):
                    if any(rp in l.lower() for rp in REPOS):
                        clean = l.rstrip(".,;")
                        found.add(clean)
                        links.setdefault(clean, [])
                        tag = r.get("doi") or r.get("pmid") or pmcid
                        if tag not in links[clean]:
                            links[clean].append(tag)
                rec["links"] = sorted(found)
        results.append(rec)
        if rec["links"]:
            print(f"  [{i}/{len(rows)}] {rec['title'][:60]:62s} 链接 {len(rec['links'])}")
        time.sleep(0.3)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dataset_mentions_from_papers.json").write_text(
        json.dumps({"n_papers": len(results), "papers": results, "links": links},
                   indent=2, ensure_ascii=False), encoding="utf-8")

    # 只保留「可能含子宫内膜数据」的仓库链接
    interesting = {l: p for l, p in links.items()
                   if any(k in l.lower() for k in ("zenodo", "figshare", "dryad", "mendeley",
                                                   "dataport", "huggingface", "kaggle", "osf",
                                                   "biostudies", "grand-challenge"))}
    lines = ["# 从论文数据可用性声明中挖到的数据链接", "",
             f"处理 OA 论文 {len(results)} 篇；命中仓库链接 {len(links)} 条，"
             f"其中数据仓库类 **{len(interesting)}** 条", "",
             "## 数据仓库链接（优先关注）", "", "| 链接 | 来源论文 |", "| --- | --- |"]
    for l, papers in sorted(interesting.items()):
        lines.append(f"| {l[:120]} | {', '.join(papers[:3])} |")
    lines += ["", "## 其他链接（代码/补充材料）", "", "| 链接 | 来源论文 |", "| --- | --- |"]
    for l, papers in sorted(links.items()):
        if l not in interesting:
            lines.append(f"| {l[:120]} | {', '.join(papers[:2])} |")
    (OUT / "dataset_mentions_from_papers.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\n数据仓库链接 {len(interesting)} 条：")
    for l in sorted(interesting)[:20]:
        print(f"  {l[:110]}")
    print(f"\n-> {OUT/'dataset_mentions_from_papers.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
