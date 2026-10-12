#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
07_probe_hf_download.py
=======================
诊断 HuggingFace 下载链路：区分「模型 gated」与「本机网络/工具问题」。
用 GET 直连 resolve URL，检查 HTTP 状态、跳转目标与响应头。

输出：data/manifests/hf_download_probe.json
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANI = ROOT / "data" / "manifests"

CASES = [
    ("phikon-v2 直链 Range", "https://huggingface.co/owkin/phikon-v2/resolve/main/model.safetensors", {"Range": "bytes=0-255"}),
    ("phikon-v2 直链无 Range", "https://huggingface.co/owkin/phikon-v2/resolve/main/model.safetensors", {}),
    ("phikon-v2 镜像", "https://hf-mirror.com/owkin/phikon-v2/resolve/main/model.safetensors", {"Range": "bytes=0-255"}),
    ("hibou-L 直链", "https://huggingface.co/histai/hibou-L/resolve/main/model.safetensors", {"Range": "bytes=0-255"}),
    ("UNI2-h 直链", "https://huggingface.co/MahmoodLab/UNI2-h/resolve/main/pytorch_model.bin", {"Range": "bytes=0-255"}),
    ("W8Yi UCEC 直链", "https://huggingface.co/datasets/W8Yi/tcga-wsi-uni2h-features/resolve/main/TCGA-UCEC/features/TCGA-2E-A9G8-01Z-00-DX1.h5", {"Range": "bytes=0-255"}),
    ("W8Yi UCEC 镜像", "https://hf-mirror.com/datasets/W8Yi/tcga-wsi-uni2h-features/resolve/main/TCGA-UCEC/features/TCGA-2E-A9G8-01Z-00-DX1.h5", {"Range": "bytes=0-255"}),
    ("TITAN 直链", "https://huggingface.co/datasets/MahmoodLab/TITAN/resolve/main/TCGA_TITAN_features.pkl", {"Range": "bytes=0-255"}),
]


def main() -> int:
    MANI.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, url, extra in CASES:
        headers = {"User-Agent": "Mozilla/5.0 (endo-wsi-research)"}
        headers.update(extra)
        rec = {"name": name, "url": url}
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = resp.read(64)
                rec.update({
                    "status": resp.status,
                    "final_url_host": urllib.parse.urlparse(resp.url).netloc if hasattr(urllib, "parse") else resp.url[:80],
                    "content_length": resp.headers.get("Content-Length"),
                    "x_linked_size": resp.headers.get("x-linked-size"),
                    "content_type": resp.headers.get("Content-Type"),
                    "head_bytes": body[:16].decode("latin-1"),
                    "ok": True,
                })
        except urllib.error.HTTPError as exc:
            rec.update({"status": exc.code, "reason": str(exc.reason), "ok": False,
                        "server": exc.headers.get("Server") if exc.headers else None,
                        "www_authenticate": exc.headers.get("WWW-Authenticate") if exc.headers else None})
        except Exception as exc:  # noqa: BLE001
            rec.update({"status": "ERR", "reason": f"{type(exc).__name__}: {exc}", "ok": False})
        rows.append(rec)
        flag = "OK  " if rec.get("ok") else "FAIL"
        size = rec.get("x_linked_size") or rec.get("content_length") or "-"
        print(f"  [{flag}] {name:24s} status={rec.get('status')} size={size} {rec.get('reason','')}")

    (MANI / "hf_download_probe.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {MANI/'hf_download_probe.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
