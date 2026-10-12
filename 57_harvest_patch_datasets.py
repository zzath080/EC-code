#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
57_harvest_patch_datasets.py
============================
系统性收集**多个来源**的开放组织病理 patch 数据集，用于构建跨来源泛化矩阵。

设计意图：
    公开数据中「有良性标签」与「是活检」从未同时出现，因此把研究问题定为可执行的版本：
        **病理基础模型的嵌入在跨来源时是否稳健？域内高 AUC 是否可信？**
    该问题需要的是「多个来源」而非「活检」，而多来源数据大多体量小、可立即下载。

检索策略：
    HuggingFace（image-classification / pathology 关键词）、Kaggle（多关键词），
    对候选打分（许可宽松、体量适中、组织病理相关、有类别目录结构），
    并支持一键下载高分候选。

用法：
    python scripts/57_harvest_patch_datasets.py --search
    python scripts/57_harvest_patch_datasets.py --search --download --top 6 --max-gb-each 1.5
输出：
    data/manifests/patch_dataset_candidates.json
    data/manifests/patch_dataset_candidates.md
    data/raw/patch_multisource/<name>/...
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "manifests"
DL = ROOT / "data" / "raw" / "patch_multisource"
UA = {"User-Agent": "Mozilla/5.0 (endo-wsi-research)"}

HF_SEARCH = ["histopathology", "pathology", "histology", "cancer histopathology",
             "tumor histopathology", "colon cancer histology", "breast cancer histology",
             "lung cancer histology", "lymph node histology"]

KAGGLE_SEARCH = ["histopathology dataset", "pathology images", "histology cancer",
                 "colon cancer histopathology", "breast cancer histopathology",
                 "lung cancer histopathology", "oral cancer histopathology",
                 "kidney histopathology", "lymphoma histopathology"]

PATHO_KW = ["histopath", "patholog", "histolog", "biopsy", "tissue", "h&e", "he-stain",
            "cancer", "tumor", "tumour", "carcinoma", "malignan", "benign", "normal"]
EXCLUDE_KW = ["mri", "ct scan", "x-ray", "radiolog", "eeg", "ecg", "text", "nlp",
              "survey", "churn", "stock", "housing", "iris", "mnist"]

PERMISSIVE = ("cc0", "cc-by", "cc by", "mit", "apache", "public domain", "open")


def jget(url: str, timeout: int = 90):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return json.load(r)


def hf_search(term: str, limit: int = 30) -> list[dict]:
    u = f"https://huggingface.co/api/datasets?search={urllib.parse.quote(term)}&limit={limit}"
    return [{"id": d.get("id"), "downloads": d.get("downloads"), "likes": d.get("likes"),
             "gated": d.get("gated"), "lastModified": d.get("lastModified"),
             "tags": d.get("tags", [])} for d in jget(u)]


def kaggle_search(term: str) -> list[dict]:
    u = f"https://www.kaggle.com/api/v1/datasets/list?search={urllib.parse.quote(term)}"
    out = []
    for x in jget(u) if isinstance(jget(u), list) else []:
        if not isinstance(x, dict):
            continue
        out.append({
            "ref": x.get("ref"),
            "title": x.get("titleNullable") or x.get("title"),
            "subtitle": x.get("subtitleNullable") or x.get("subtitle"),
            "bytes": x.get("totalBytesNullable") or x.get("totalBytes") or 0,
            "license": x.get("licenseNameNullable") or x.get("licenseName"),
            "downloads": x.get("downloadCount"),
            "votes": x.get("voteCount"),
            "usability": x.get("usabilityRating"),
        })
    return out


def score_kaggle(c: dict) -> tuple[int, list[str]]:
    blob = f"{c.get('title')} {c.get('subtitle')}".lower()
    s, why = 0, []
    if any(k in blob for k in PATHO_KW):
        s += 3; why.append("组织病理相关")
    if any(k in blob for k in EXCLUDE_KW):
        s -= 5; why.append("疑似无关模态")
    lic = str(c.get("license") or "").lower()
    if any(p in lic for p in PERMISSIVE):
        s += 2; why.append("许可宽松")
    b = c.get("bytes") or 0
    if 5e6 <= b <= 3e9:
        s += 2; why.append("体量适中")
    elif b > 3e9:
        s -= 1; why.append("体量偏大")
    if (c.get("downloads") or 0) > 20:
        s += 1; why.append("下载量较高")
    if (c.get("usability") or 0) > 0.5:
        s += 1; why.append("可用性评分较好")
    return s, why


def download_kaggle(ref: str, dest_name: str) -> tuple[bool, str]:
    """复用 39_fetch_kaggle_zip.py 的并发分段下载器。"""
    zip_path = DL / f"{dest_name}.zip"
    cmd = [sys.executable, str(ROOT / "scripts" / "39_fetch_kaggle_zip.py"),
           "--ref", ref, "--segments", "8", "--workers", "8",
           "--out", str((DL).relative_to(ROOT))]
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), timeout=3600)
    ok = p.returncode == 0
    return ok, (p.stdout or "")[-400:] + (p.stderr or "")[-200:]


def unzip(zip_path: Path, dest: Path) -> int:
    import zipfile
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
        return len(z.namelist())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--top", type=int, default=6)
    ap.add_argument("--max-gb-each", type=float, default=1.5)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    hf: dict[str, list] = {}
    kg: dict[str, list] = {}

    print("=== HuggingFace 检索 ===")
    for t in HF_SEARCH:
        try:
            rows = hf_search(t)
            hf[t] = rows
            hit = [r for r in rows if any(k in (r["id"] or "").lower() for k in PATHO_KW)]
            print(f"  [{t}] 命中 {len(rows)}，病理相关 {len(hit)}")
            for r in hit[:4]:
                print(f"      {r['id']}  gated={r['gated']} dl={r['downloads']}")
        except Exception as exc:  # noqa: BLE001
            print(f"  [{t}] 失败 {type(exc).__name__}")

    print("\n=== Kaggle 检索 ===")
    for t in KAGGLE_SEARCH:
        try:
            rows = kaggle_search(t)
            kg[t] = rows
            print(f"  [{t}] 命中 {len(rows)}")
        except Exception as exc:  # noqa: BLE001
            print(f"  [{t}] 失败 {type(exc).__name__}")

    # 去重 + 打分
    cands: dict[str, dict] = {}
    for t, rows in kg.items():
        for c in rows:
            ref = c.get("ref")
            if not ref:
                continue
            if ref in cands:
                cands[ref]["matched"].append(t)
            else:
                s, why = score_kaggle(c)
                cands[ref] = {**c, "score": s, "why": why, "matched": [t]}

    ranked = sorted(cands.values(), key=lambda c: -c["score"])
    (OUT / "patch_dataset_candidates.json").write_text(
        json.dumps({"huggingface": hf, "kaggle": cands}, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["# 跨来源组织病理 patch 数据集候选", "",
             "评分维度：组织病理相关性(+3)、许可宽松(+2)、体量适中(+2)、下载量(+1)、可用性(+1)、",
             "无关模态(-5)、体量过大(-1)", "",
             "| 分数 | ref | 标题 | 体量(MB) | 许可 | 下载 | 理由 |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for c in ranked[:40]:
        lines.append(f"| {c['score']} | {c['ref']} | {str(c['title'])[:44]} | "
                     f"{(c['bytes'] or 0)/1024**2:.0f} | {str(c['license'])[:16]} | "
                     f"{c['downloads']} | {','.join(c['why'])} |")
    (OUT / "patch_dataset_candidates.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\n候选 {len(ranked)} 个；前 15：")
    for c in ranked[:15]:
        print(f"  [{c['score']:2d}] {c['ref'][:52]:54s} {(c['bytes'] or 0)/1024**2:7.0f} MB  "
              f"{str(c['license'])[:14]:16s} {','.join(c['why'])}")

    if not args.download:
        print(f"\n-> {OUT/'patch_dataset_candidates.md'}")
        print("加 --download 可下载高分候选")
        return 0

    # 下载：跳过已存在的、跳过过大的
    picked = [c for c in ranked
              if c["score"] >= 5 and 0 < (c["bytes"] or 0) <= args.max_gb_each * 1024 ** 3][: args.top]
    print(f"\n=== 下载 {len(picked)} 个候选 ===")
    DL.mkdir(parents=True, exist_ok=True)
    results = []
    for c in picked:
        name = re.sub(r"[^A-Za-z0-9_]+", "_", c["ref"].split("/")[-1])[:40]
        target_dir = DL / name
        if target_dir.exists() and any(target_dir.rglob("*.jpg")):
            print(f"  [SKIP] {name} 已解压")
            continue
        print(f"  [{c['score']}] {c['ref']} -> {name}")
        ok, msg = download_kaggle(c["ref"], name)
        print(f"      {'OK' if ok else 'FAIL'} {msg[:200]}")
        zp = DL / f"{name}.zip"
        n = 0
        if zp.exists():
            try:
                n = unzip(zp, target_dir)
                print(f"      解压 {n} 个条目")
                zp.unlink()   # 解压后删 zip 省空间
            except Exception as exc:  # noqa: BLE001
                print(f"      解压失败 {type(exc).__name__}: {exc}")
        results.append({"ref": c["ref"], "name": name, "ok": ok, "entries": n,
                        "license": c["license"], "bytes": c["bytes"]})

    (OUT / "patch_dataset_download_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {OUT/'patch_dataset_download_results.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
