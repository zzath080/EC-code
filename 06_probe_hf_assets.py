#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
06_probe_hf_assets.py
=====================
用 GET（Range: bytes=0-255）而非 HEAD 探测 HuggingFace 资产是否真的可匿名访问。
HEAD 在 HF CDN 上会误报 403，必须用带 Range 的 GET 才能得到可信结论。

输出：data/manifests/hf_asset_probe_get.json
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANI = ROOT / "data" / "manifests"
HF = "https://huggingface.co"

TARGETS = [
    ("dataset", "W8Yi/tcga-wsi-uni2h-features", "TCGA-UCEC/features/TCGA-2E-A9G8-01Z-00-DX1.h5"),
    ("dataset", "W8Yi/tcga-wsi-uni2h-features", "README.md"),
    ("dataset", "MahmoodLab/TITAN", "TCGA_TITAN_features.pkl"),
    ("dataset", "MahmoodLab/UNI2-h-features", "TCGA/TCGA-UCEC.tar.gz"),
    ("model", "owkin/phikon-v2", "model.safetensors"),
    ("model", "owkin/phikon", "pytorch_model.bin"),
    ("model", "histai/hibou-L", "model.safetensors"),
    ("model", "histai/hibou-b", "model.safetensors"),
    ("model", "MahmoodLab/UNI2-h", "pytorch_model.bin"),
    ("model", "MahmoodLab/CONCH", "pytorch_model.bin"),
    ("model", "paige-ai/Virchow2", "model.safetensors"),
    ("model", "bioptimus/H-optimus-0", "pytorch_model.bin"),
    ("model", "google/vit-base-patch16-224", "model.safetensors"),
    ("model", "timm/vit_large_patch16_224.augreg_in21k", "pytorch_model.bin"),
]


def probe(prefix: str, repo: str, path: str) -> dict:
    base = f"{HF}/{prefix}/{repo}/resolve/main/{path}"
    req = urllib.request.Request(base, headers={
        "User-Agent": "Mozilla/5.0 (endo-wsi-research)",
        "Range": "bytes=0-255",
    })
    rec = {"prefix": prefix, "repo": repo, "path": path, "url": base}
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            body = resp.read(256)
            rec.update({
                "accessible": True,
                "status": resp.status,
                "full_size": resp.headers.get("x-linked-size") or resp.headers.get("Content-Length"),
                "content_type": resp.headers.get("Content-Type"),
                "first_bytes": body[:24].decode("latin-1"),
            })
    except urllib.error.HTTPError as exc:
        rec.update({"accessible": False, "status": exc.code, "note": exc.reason,
                    "gated": exc.code in (401, 403)})
    except Exception as exc:  # noqa: BLE001
        rec.update({"accessible": False, "status": "ERR", "note": f"{type(exc).__name__}: {exc}"})
    return rec


def main() -> int:
    MANI.mkdir(parents=True, exist_ok=True)
    rows = []
    for prefix, repo, path in TARGETS:
        r = probe(prefix, repo, path)
        rows.append(r)
        size = r.get("full_size")
        tag = "OK  " if r["accessible"] else "BLOCK"
        sz = f"{int(size)/1024**2:.1f} MB" if size and str(size).isdigit() else str(size)
        print(f"  [{tag}] {prefix:7s} {repo:45s} {str(path):45s} {sz:>12s} {r.get('note','')}")

    (MANI / "hf_asset_probe_get.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    ok = [r for r in rows if r["accessible"]]
    print(f"\n可匿名访问 {len(ok)}/{len(rows)}")
    print(f"-> {MANI/'hf_asset_probe_get.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
