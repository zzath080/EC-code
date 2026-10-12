#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
15_uni2h_manifest.py
====================
为 HuggingFace 上的 W8Yi/tcga-wsi-uni2h-features 建立可复现的下载清单。

背景（已实测）：
    该仓库公开、无需机构邮箱即可匿名下载（HTTP 206），包含 TCGA 各癌种的
    UNI2-h patch 特征 H5，其中 TCGA-UCEC 覆盖 566 张诊断切片。
    schema: features[1,N,1536] / coords[1,N,2] / annots[1,N,1]（UNI2-h, 256px@20x）

用法：
    python scripts/15_uni2h_manifest.py                     # 只统计 UCEC
    python scripts/15_uni2h_manifest.py --project TCGA-BRCA_IDC
    python scripts/15_uni2h_manifest.py --all-projects
输出：
    data/manifests/uni2h/<project>_features.csv
    data/manifests/uni2h_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "manifests" / "uni2h"
REPO = "W8Yi/tcga-wsi-uni2h-features"
API = "https://huggingface.co/api/datasets/{repo}/tree/main/{path}"
DL = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"
UA = {"User-Agent": "endo-wsi-research/1.0"}


def tree(path: str = "") -> list[dict]:
    url = API.format(repo=REPO, path=path)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=90) as resp:
        return json.load(resp)


def project_files(project: str) -> list[dict]:
    """列出某个 TCGA 项目下的全部 .h5 特征文件。"""
    out: list[dict] = []
    for sub in ("features", "vis"):
        try:
            entries = tree(f"{project}/{sub}")
        except Exception:  # noqa: BLE001
            continue
        for e in entries:
            if e.get("type") == "file":
                out.append({
                    "project": project,
                    "kind": sub,
                    "path": e["path"],
                    "name": Path(e["path"]).name,
                    "size": int(e.get("size") or 0),
                    "url": DL.format(repo=REPO, path=e["path"]),
                })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="TCGA-UCEC")
    ap.add_argument("--all-projects", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    root = tree("")
    projects = [e["path"] for e in root if e.get("type") == "directory"]
    print(f"仓库 {REPO} 共 {len(projects)} 个项目目录")

    targets = projects if args.all_projects else [args.project]
    summary: dict[str, dict] = {}

    for proj in targets:
        rows = project_files(proj)
        h5 = [r for r in rows if r["kind"] == "features" and r["name"].endswith(".h5")]
        total = sum(r["size"] for r in h5)
        summary[proj] = {"n_features": len(h5), "bytes": total, "gb": round(total / 1024 ** 3, 2)}
        if h5:
            with open(OUT / f"{proj}_features.csv", "w", encoding="utf-8-sig", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=["project", "kind", "path", "name", "size", "url"])
                w.writeheader()
                w.writerows(h5)
        print(f"  {proj:22s} 特征文件 {len(h5):4d}  合计 {total/1024**3:7.2f} GB")
        sys.stdout.flush()

    (ROOT / "data" / "manifests" / "uni2h_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if not args.all_projects:
        p = summary.get(args.project, {})
        print(f"\n{args.project}: {p.get('n_features', 0)} 个特征文件，{p.get('gb', 0)} GB")
        print(f"清单 -> {OUT / (args.project + '_features.csv')}")
    else:
        grand = sum(v["bytes"] for v in summary.values())
        print(f"\n全部项目合计 {sum(v['n_features'] for v in summary.values())} 个文件，{grand/1024**3:.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
