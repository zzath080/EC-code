#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
05_fetch_pretrained_features.py
===============================
免 GPU、免 WSI 解码即可开始的"零成本"资产获取。

包括：
    1. W8Yi/tcga-wsi-uni2h-features  —— TCGA-UCEC 566 张诊断切片的 UNI2-h patch 特征 (H5, 公开无 gated)
    2. MahmoodLab/TITAN               —— 全 TCGA slide 级特征 (pkl, gated)
    3. HuggingFace 模型权重可用性探测（UNI2-h / CONCH / Hibou / Phikon 等是否可匿名拉取）

用法：
    python scripts/05_fetch_pretrained_features.py --probe-only
    python scripts/05_fetch_pretrained_features.py --fetch-titan
    python scripts/05_fetch_pretrained_features.py --list-uni2h
    python scripts/05_fetch_pretrained_features.py --fetch-uni2h --max-files 20
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "data" / "features"
MANI = ROOT / "data" / "manifests"

# HuggingFace 上值得探测的模型/特征仓库
PROBES = [
    ("模型", "MahmoodLab/UNI", "pytorch_model.bin", "gated"),
    ("模型", "MahmoodLab/UNI2-h", "pytorch_model.bin", "gated"),
    ("模型", "MahmoodLab/CONCH", "pytorch_model.bin", "gated"),
    ("模型", "bioptimus/H-optimus-0", "pytorch_model.bin", "gated"),
    ("模型", "bioptimus/H-optimus-1", "pytorch_model.bin", "gated"),
    ("模型", "paige-ai/Virchow2", "model.safetensors", "gated"),
    ("模型", "prov-gigapath/prov-gigapath", "pytorch_model.bin", "gated"),
    ("模型", "owkin/phikon-v2", "model.safetensors", "open"),
    ("模型", "owkin/phikon", "pytorch_model.bin", "open"),
    ("模型", "1aurent/vit_large_patch16_384.owkin_pancancer", "model.safetensors", "open"),
    ("模型", "histai/hibou-L", "model.safetensors", "open"),
    ("模型", "histai/hibou-b", "model.safetensors", "open"),
    ("模型", "google/vit-base-patch16-224", "model.safetensors", "open"),
    ("特征", "W8Yi/tcga-wsi-uni2h-features", "TCGA-UCEC/features", "open"),
    ("特征", "MahmoodLab/TITAN", "TCGA_TITAN_features.pkl", "gated"),
    ("特征", "MahmoodLab/UNI2-h-features", "TCGA/TCGA-UCEC.tar.gz", "gated"),
]

HF = "https://huggingface.co"


def probe(repo: str, path: str) -> dict:
    """探测一个 HF 仓库文件是否可匿名访问（只用 HEAD，不下载）。"""
    url = f"{HF}/{repo}/resolve/main/{path}"
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "endo-wsi-research/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            size = resp.headers.get("Content-Length") or resp.headers.get("x-linked-size") or "?"
            return {"repo": repo, "path": path, "status": resp.status, "size": size,
                    "accessible": True, "final_url": resp.url}
    except urllib.error.HTTPError as exc:
        return {"repo": repo, "path": path, "status": exc.code, "accessible": False,
                "note": "gated/需授权" if exc.code in (401, 403) else exc.reason}
    except Exception as exc:  # noqa: BLE001
        return {"repo": repo, "path": path, "status": "ERR", "accessible": False,
                "note": f"{type(exc).__name__}: {exc}"}


def hf_tree(repo: str, path: str = "") -> list[dict]:
    url = f"{HF}/api/datasets/{repo}/tree/main/{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "endo-wsi-research/1.0"})
    with urllib.request.urlopen(req, timeout=90) as resp:
        return json.load(resp)


def download(url: str, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "endo-wsi-research/1.0"})
    with urllib.request.urlopen(req, timeout=600) as resp, open(part, "wb") as fh:
        while True:
            chunk = resp.read(4 * 1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
    os.replace(part, dest)
    return f"{dest.stat().st_size/1024**2:.1f} MB"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe-only", action="store_true")
    ap.add_argument("--list-uni2h", action="store_true")
    ap.add_argument("--fetch-uni2h", action="store_true")
    ap.add_argument("--fetch-titan", action="store_true")
    ap.add_argument("--max-files", type=int, default=None)
    args = ap.parse_args()

    FEAT.mkdir(parents=True, exist_ok=True)
    MANI.mkdir(parents=True, exist_ok=True)

    print("=== HuggingFace 资产可用性探测 ===")
    results = []
    for kind, repo, path, expect in PROBES:
        r = probe(repo, path)
        r["kind"] = kind
        r["expect"] = expect
        results.append(r)
        flag = "OK  " if r["accessible"] else "BLOCK"
        print(f"  [{flag}] {kind} {repo:55s} {str(r.get('size')):>14s}  {r.get('note','')}")
    (MANI / "hf_asset_probe.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  -> {MANI/'hf_asset_probe.json'}")

    if args.list_uni2h:
        print("\n=== W8Yi/tcga-wsi-uni2h-features 目录 ===")
        entries = hf_tree("W8Yi/tcga-wsi-uni2h-features")
        for e in entries:
            print(f"  {e.get('type'):5s} {e.get('path')}")
        try:
            sub = hf_tree("W8Yi/tcga-wsi-uni2h-features", "TCGA-UCEC")
            for e in sub:
                print(f"    {e.get('type'):5s} {e.get('path')}  {e.get('size')}")
        except Exception as exc:  # noqa: BLE001
            print("  子目录列举失败:", exc)

    if args.fetch_uni2h:
        print("\n=== 拉取 UNI2-h 特征清单 ===")
        sub = hf_tree("W8Yi/tcga-wsi-uni2h-features", "TCGA-UCEC/features")
        rows = [e for e in sub if e.get("type") == "file" and e["path"].endswith(".h5")]
        total = sum(e.get("size", 0) for e in rows)
        print(f"  UCEC 特征文件 {len(rows)} 个，合计 {total/1024**3:.2f} GB")
        import csv
        with open(MANI / "uni2h_ucec_features.csv", "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["path", "size_bytes", "url"])
            for e in rows:
                w.writerow([e["path"], e.get("size", 0),
                            f"{HF}/datasets/W8Yi/tcga-wsi-uni2h-features/resolve/main/{e['path']}"])
        print(f"  -> {MANI/'uni2h_ucec_features.csv'}")

        if args.max_files:
            rows.sort(key=lambda e: e.get("size", 0))
            for e in rows[: args.max_files]:
                url = f"{HF}/datasets/W8Yi/tcga-wsi-uni2h-features/resolve/main/{e['path']}"
                dest = FEAT / "uni2h" / "ucec" / Path(e["path"]).name
                try:
                    info = download(url, dest)
                    print(f"  [OK] {Path(e['path']).name}  {info}")
                except Exception as exc:  # noqa: BLE001
                    print(f"  [FAIL] {e['path']}: {type(exc).__name__}: {exc}")

    if args.fetch_titan:
        print("\n=== 拉取 TITAN slide 级特征 ===")
        for repo, path in [("MahmoodLab/TITAN", "TCGA_TITAN_features.pkl")]:
            url = f"{HF}/{repo}/resolve/main/{path}"
            dest = FEAT / "titan" / Path(path).name
            try:
                print(f"  [OK] {path}  {download(url, dest)}")
            except Exception as exc:  # noqa: BLE001
                print(f"  [FAIL] {path}: {type(exc).__name__}: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
