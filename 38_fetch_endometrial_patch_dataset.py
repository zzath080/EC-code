#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
38_fetch_endometrial_patch_dataset.py
=====================================
获取 patch 级的子宫内膜组织病理数据集（含正常/息肉/增生/腺癌四类）。

这是本项目中**同时提供良性与恶性**的子宫内膜 patch 级数据，
用作 (a) 零成本 sanity check、(b) 良恶性任务的补充证据、(c) 特征可分性验证。

数据来源（多处尝试，任一成功即止）：
    1. Kaggle 镜像（匿名 API 可列出，下载可能需凭据）
    2. HuggingFace 上的等价镜像

原始来源：Sun H, Zeng X, Xu T, Peng G, Ma Y. Computer-aided diagnosis in histopathological
images of the endometrium using a CNN and attention mechanisms. IEEE J Biomed Health Inform.
2020;26(6):1664-1676. doi:10.1109/JBHI.2019.2944977（arXiv:1904.10626）

用法：
    python scripts/38_fetch_endometrial_patch_dataset.py --search
    python scripts/38_fetch_endometrial_patch_dataset.py --try-download
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "endometrial_patch"
UA = {"User-Agent": "Mozilla/5.0 (research)"}


def get_json(url: str):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def search_kaggle(term: str = "endometrial") -> list[dict]:
    try:
        data = get_json(f"https://www.kaggle.com/api/v1/datasets/list?search={term}")
        out = []
        for d in data:
            if isinstance(d, dict):
                out.append(d)
            else:
                out.append({"ref": str(d)})
        return out
    except Exception as exc:  # noqa: BLE001
        print(f"  Kaggle 搜索失败：{type(exc).__name__}: {exc}")
        return []


def kaggle_detail(ref: str) -> dict | None:
    try:
        return get_json(f"https://www.kaggle.com/api/v1/datasets/view/{ref}")
    except Exception as exc:  # noqa: BLE001
        print(f"  Kaggle 详情失败 {ref}：{type(exc).__name__}: {exc}")
        return None


def try_download(ref: str, dest: Path) -> bool:
    url = f"https://www.kaggle.com/api/v1/datasets/download/{ref}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=300) as resp, open(part, "wb") as fh:
            total = 0
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
                total += len(chunk)
        part.replace(dest)
        print(f"  下载成功 {dest.name}  {total/1024**2:.1f} MB")
        return True
    except urllib.error.HTTPError as exc:
        print(f"  下载被拒（HTTP {exc.code}）：Kaggle 需要凭据，跳过")
    except Exception as exc:  # noqa: BLE001
        print(f"  下载失败：{type(exc).__name__}: {exc}")
    part.unlink(missing_ok=True)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--try-download", action="store_true")
    args = ap.parse_args()

    print("=== Kaggle 搜索 'endometrial' ===")
    hits = search_kaggle("endometrial")
    print(f"返回 {len(hits)} 条")
    for h in hits[:12]:
        print(f"  {str(h)[:200]}")

    if args.search:
        return 0

    # 已知的镜像名（若存在）
    candidates = [
        "samihamuntahamahin/endometrial-histopathology",
        "obulisainaren/endometrial-cancer",
    ]
    for ref in candidates:
        d = kaggle_detail(ref)
        if d:
            print(f"\n=== {ref} ===")
            for k in ("title", "subtitle", "totalBytes", "licenseName", "lastUpdated"):
                if k in d:
                    print(f"  {k}: {d[k]}")
            files = d.get("resources") or d.get("files") or []
            print(f"  文件数: {len(files) if isinstance(files, list) else 'n/a'}")

    if args.try_download:
        for ref in candidates:
            print(f"\n尝试下载 {ref} …")
            if try_download(ref, OUT / f"{ref.split('/')[-1]}.zip"):
                break
    return 0


if __name__ == "__main__":
    sys.exit(main())
