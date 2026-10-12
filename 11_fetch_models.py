#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
11_fetch_models.py
==================
下载开放权重的基础模型到工作区（不依赖 huggingface_hub，避免额外依赖问题）。

已实测可匿名下载（HTTP 206/200）：
    - owkin/phikon-v2   : ViT-L/14, 1.21 GB, DINOv2 自监督, 160k+ WSI, 20x, 0.5 µm/px  ← 主用
    - google/vit-base-patch16-224 : ImageNet 基线对照
需授权（gated，跳过并记录）：
    - MahmoodLab/UNI2-h, MahmoodLab/CONCH, histai/hibou-L, paige-ai/Virchow2, MahmoodLab/TITAN

用法：
    python scripts/11_fetch_models.py --model phikon-v2
    python scripts/11_fetch_models.py --list
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
HF = "https://huggingface.co"

OPEN_MODELS = {
    "phikon-v2": {
        "repo": "owkin/phikon-v2",
        "kind": "models",
        "files": ["config.json", "preprocessor_config.json", "model.safetensors", "README.md"],
        "note": "ViT-L/14, DINOv2 自监督, 160k+ WSI, 20x / 0.5 µm/px",
    },
    "vit-base-in": {
        "repo": "google/vit-base-patch16-224",
        "kind": "models",
        "files": ["config.json", "preprocessor_config.json", "model.safetensors"],
        "note": "ImageNet 监督基线",
    },
}

GATED = ["MahmoodLab/UNI2-h", "MahmoodLab/CONCH", "histai/hibou-L",
         "paige-ai/Virchow2", "bioptimus/H-optimus-0", "MahmoodLab/TITAN"]


def list_repo(repo: str, kind: str = "models") -> list[dict]:
    url = f"{HF}/api/{kind}/{repo}/tree/main"
    req = urllib.request.Request(url, headers={"User-Agent": "endo-wsi-research/1.0"})
    with urllib.request.urlopen(req, timeout=90) as resp:
        return json.load(resp)


def hf_prefix(kind: str) -> str:
    """HF 直链的路径前缀：模型是 <host>/<repo>/resolve/...，数据集才是 /datasets/。"""
    return "" if kind == "models" else "/datasets"


def remote_size(url: str) -> int:
    """HEAD 取远端大小；取不到返回 0。用于识别被截断的下载。"""
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "endo-wsi-research/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            return int(resp.headers.get("Content-Length") or resp.headers.get("x-linked-size") or 0)
    except Exception:  # noqa: BLE001
        return 0


def stream_download(url: str, dest: Path, headers: dict, timeout: int = 45,
                    progress_every: int = 128 * 1024 * 1024) -> int:
    """带「读进度超时」的下载：socket 每 timeout 秒必须有数据到达，否则抛错重试。

    仅设 urlopen(timeout=...) 是不够的——它只约束连接建立与单次 recv，
    某些代理会让 recv 长期阻塞，表现为「文件大小停在某处不动」。
    """
    import socket
    total = 0
    last_report = 0
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        with open(dest, "ab" if headers.get("Range") else "wb") as fh:
            while True:
                try:
                    chunk = resp.read(1024 * 1024)
                except (socket.timeout, TimeoutError) as exc:
                    raise TimeoutError(f"读超时（{timeout}s 无数据）") from exc
                if not chunk:
                    break
                fh.write(chunk)
                total += len(chunk)
                if total - last_report >= progress_every:
                    last_report = total
                    print(f"      ...{total/1024**2:.1f} MB", flush=True)
    return total


def download_file(repo: str, kind: str, name: str, dest: Path) -> tuple[str, int]:
    url = f"{HF}{hf_prefix(kind)}/{repo}/resolve/main/{name}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    want = remote_size(url)

    if dest.exists() and dest.stat().st_size > 0:
        have = dest.stat().st_size
        if not want or have == want:
            return "cached", have
        print(f"      缓存不完整 {have/1024**2:.1f}/{want/1024**2:.1f} MB，重新下载")
        dest.unlink()

    part = dest.with_suffix(dest.suffix + ".part")
    # HF 长连接在本机会被反复掐断，靠 Range 续传啃完，故重试次数给足
    for attempt in range(1, 201):
        have = part.stat().st_size if part.exists() else 0
        if want and have > want:
            part.unlink(missing_ok=True)
            have = 0
        if want and have == want:
            break
        headers = {"User-Agent": "endo-wsi-research/1.0"}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            stream_download(url, part, headers)
            if attempt % 10 == 0:
                now = part.stat().st_size if part.exists() else 0
                print(f"      第 {attempt} 次尝试后 {now/1024**2:.1f}"
                      f"{f'/{want/1024**2:.1f}' if want else ''} MB", flush=True)
        except Exception as exc:  # noqa: BLE001
            if attempt % 5 == 1:
                now = part.stat().st_size if part.exists() else 0
                print(f"      续传({attempt}) {now/1024**2:.1f}"
                      f"{f'/{want/1024**2:.1f}' if want else ''} MB: {type(exc).__name__}", flush=True)

    got = part.stat().st_size if part.exists() else 0
    if want and got != want:
        raise RuntimeError(f"下载不完整：{got} / {want} 字节")
    os.replace(part, dest)
    return "ok", dest.stat().st_size


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="phikon-v2")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for alias, spec in OPEN_MODELS.items():
            print(f"[开放] {alias:14s} {spec['repo']:35s} {spec['note']}")
            try:
                for e in list_repo(spec["repo"], spec["kind"]):
                    print(f"          {e.get('type'):5s} {e.get('path'):40s} {e.get('size')}")
            except Exception as exc:  # noqa: BLE001
                print(f"          列举失败: {exc}")
        print("\n[需授权，本次跳过]")
        for g in GATED:
            print(f"  {g}")
        return 0

    spec = OPEN_MODELS.get(args.model)
    if not spec:
        print(f"未知模型 {args.model}；可用：{list(OPEN_MODELS)}", file=sys.stderr)
        return 2

    dest_root = MODELS / args.model
    print(f"下载 {args.model} ({spec['repo']}) -> {dest_root}")
    for name in spec["files"]:
        try:
            status, size = download_file(spec["repo"], spec["kind"], name, dest_root / name)
            print(f"  [{status:6s}] {name:32s} {size/1024**2:8.1f} MB")
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL  ] {name}: {type(exc).__name__}: {exc}")
    print(f"\n-> {dest_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
