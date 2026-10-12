#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
70_fetch_second_encoder.py
==========================
第二编码器对照实验的前置：实测并下载**开放权重**的第二个病理基础模型。

**不使用任何 gated 模型**（UNI2-h / CONCH / Virchow2 / H-optimus / Hibou 等需机构邮箱授权，
本实验一律不用）。

候选（按优先级）：
    1. `owkin/phikon`        —— Phikon-v2 的前代（iBOT 自监督，在 TCGA 上训练），
                                与 v2 架构与训练数据均不同，是真正独立的对照
    2. `1aurent/vit_large_patch16_384.owkin_pancancer` —— 另一条独立路线（若可用）
    3. `google/vit-base-patch16-224` —— ImageNet 监督基线（非病理模型，仅作下界参考）

用法：
    python scripts/70_fetch_second_encoder.py --probe
    python scripts/70_fetch_second_encoder.py --fetch phikon
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
MANI = ROOT / "data" / "manifests"
HF = "https://huggingface.co"
UA = {"User-Agent": "Mozilla/5.0 (endo-wsi-research)"}

CANDIDATES = {
    "phikon": {
        "repo": "owkin/phikon",
        "files": ["config.json", "preprocessor_config.json", "pytorch_model.bin"],
        "note": "Phikon v1：iBOT 自监督，40M TCGA 图像块；Phikon-v2 的前代",
    },
    "vit-large-pancancer": {
        "repo": "1aurent/vit_large_patch16_384.owkin_pancancer",
        "files": ["config.json", "preprocessor_config.json", "model.safetensors"],
        "note": "Owkin 泛癌 ViT-L/16 384（另一条独立路线）",
    },
    "vit-base-in": {
        "repo": "google/vit-base-patch16-224",
        "files": ["config.json", "preprocessor_config.json", "model.safetensors"],
        "note": "ImageNet 监督基线（非病理模型）",
    },
}

GATED_NOTE = ("以下模型为 gated，**本实验不使用**：MahmoodLab/UNI2-h、MahmoodLab/CONCH、"
              "paige-ai/Virchow2、bioptimus/H-optimus-0、histai/hibou-L、histai/hibou-b")


def probe(repo: str, fname: str) -> dict:
    url = f"{HF}/{repo}/resolve/main/{fname}"
    try:
        req = urllib.request.Request(url, headers={**UA, "Range": "bytes=0-255"})
        with urllib.request.urlopen(req, timeout=60) as r:
            size = r.headers.get("x-linked-size") or r.headers.get("Content-Length")
            return {"file": fname, "ok": True, "status": r.status, "size": size}
    except urllib.error.HTTPError as exc:
        return {"file": fname, "ok": False, "status": exc.code,
                "gated": exc.code in (401, 403)}
    except Exception as exc:  # noqa: BLE001
        return {"file": fname, "ok": False, "error": type(exc).__name__}


def remote_size(url: str) -> int:
    try:
        req = urllib.request.Request(url, method="HEAD", headers=UA)
        with urllib.request.urlopen(req, timeout=60) as r:
            return int(r.headers.get("Content-Length") or r.headers.get("x-linked-size") or 0)
    except Exception:  # noqa: BLE001
        return 0


def download(repo: str, fname: str, dest: Path) -> tuple[str, int]:
    import time
    url = f"{HF}/{repo}/resolve/main/{fname}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    want = remote_size(url)
    if dest.exists() and (not want or dest.stat().st_size == want):
        return "cached", dest.stat().st_size
    part = dest.with_suffix(dest.suffix + ".part")
    # 本机对大文件的长连接会被反复掐断（实测每十几 MB 断一次），
    # 故重试上限给足，靠 Range 续传把文件啃完。
    for attempt in range(1, 801):
        have = part.stat().st_size if part.exists() else 0
        if want and have > want:
            part.unlink(missing_ok=True)
            have = 0
        if want and have == want:
            break
        headers = dict(UA)
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=45) as r:
                mode = "ab" if (have and r.status == 206) else "wb"
                if mode == "wb":
                    have = 0
                with open(part, mode) as fh:
                    while True:
                        c = r.read(1024 * 1024)
                        if not c:
                            break
                        fh.write(c)
            if attempt % 25 == 1:
                now = part.stat().st_size if part.exists() else 0
                print(f"    续传 {attempt}: {now/1024**2:.1f}/{want/1024**2:.1f} MB", flush=True)
        except Exception:  # noqa: BLE001
            time.sleep(min(20, 1 + attempt // 5))
            continue
    got = part.stat().st_size if part.exists() else 0
    if want and got != want:
        raise RuntimeError(f"下载不完整 {got}/{want}")
    os.replace(part, dest)
    return "ok", dest.stat().st_size


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--fetch", default=None, help="要下载的候选别名")
    args = ap.parse_args()

    MANI.mkdir(parents=True, exist_ok=True)
    report = {}
    if args.probe or not args.fetch:
        print("=== 开放权重候选实测（HTTP Range 探测）===")
        for alias, spec in CANDIDATES.items():
            print(f"\n[{alias}] {spec['repo']}  —— {spec['note']}")
            rows = []
            for f in spec["files"]:
                r = probe(spec["repo"], f)
                rows.append(r)
                flag = "OK   " if r.get("ok") else ("GATED " if r.get("gated") else "FAIL  ")
                size = r.get("size")
                sz = f"{int(size)/1024**2:.1f} MB" if size and str(size).isdigit() else str(size)
                print(f"   [{flag}] {f:26s} {sz}")
            report[alias] = rows
        (MANI / "second_encoder_probe.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n{GATED_NOTE}")
        print(f"\n-> {MANI/'second_encoder_probe.json'}")

    if args.fetch:
        spec = CANDIDATES.get(args.fetch)
        if not spec:
            print(f"[ERROR] 未知别名 {args.fetch}；可选 {list(CANDIDATES)}", file=sys.stderr)
            return 2
        dest_root = MODELS / args.fetch
        print(f"\n下载 {args.fetch} ({spec['repo']}) -> {dest_root}")
        for f in spec["files"]:
            try:
                st, size = download(spec["repo"], f, dest_root / f)
                print(f"  [{st:6s}] {f:26s} {size/1024**2:8.1f} MB")
            except Exception as exc:  # noqa: BLE001
                print(f"  [FAIL  ] {f}: {type(exc).__name__}: {exc}")
        print(f"\n-> {dest_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
