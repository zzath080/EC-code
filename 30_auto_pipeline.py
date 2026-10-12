#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
30_auto_pipeline.py
===================
自动补全流水线：反复扫描本地切片，对缺 patch 索引的做分块、对缺特征的提特征，
直到没有可处理的新切片为止。适合在长时间下载期间反复调用。

用法：
    python scripts/30_auto_pipeline.py                 # 一轮：补索引 + 补特征
    python scripts/30_auto_pipeline.py --rounds 3      # 最多 3 轮（每轮之间等待）
    python scripts/30_auto_pipeline.py --only patches  # 只补索引
    python scripts/30_auto_pipeline.py --encoder phikon-v2
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
WSI_EXTS = ("*.svs", "*.tif", "*.tiff", "*.ndpi")


def sid_of(name: str) -> str:
    s = re.sub(r"\.(svs|h5|tif|tiff|ndpi)$", "", name, flags=re.I)
    return re.sub(r"\.[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$", "", s)


def local_slides() -> dict[str, Path]:
    out: dict[str, Path] = {}
    for d in [ROOT / "TCGA-UCEC", ROOT / "data" / "raw"]:
        if not d.exists():
            continue
        for ext in WSI_EXTS:
            for p in d.rglob(ext):
                out.setdefault(sid_of(p.name), p)
    return out


def patched() -> set[str]:
    out: set[str] = set()
    for p in (ROOT / "data" / "patches").rglob("*.csv"):
        out.add(p.stem)
        out.add(sid_of(p.stem))
    return out


def featured() -> set[str]:
    out: set[str] = set()
    for p in (ROOT / "data" / "features").rglob("*.h5"):
        if "_flat" in str(p):
            continue
        out.add(p.stem)
        out.add(sid_of(p.stem))
    return out


def run(cmd: list[str]) -> int:
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run(cmd, cwd=str(ROOT)).returncode


def tile_new(limit: int | None = None) -> int:
    """对缺索引的切片分块。按目录分组以复用 --input。"""
    local = local_slides()
    done = patched()
    todo = {sid: p for sid, p in local.items() if sid not in done}
    if not todo:
        print("没有需要分块的切片")
        return 0
    if limit:
        todo = dict(list(todo.items())[:limit])
    print(f"需要分块：{len(todo)} 张")

    by_dir: dict[Path, int] = {}
    for p in todo.values():
        by_dir[p.parent] = by_dir.get(p.parent, 0) + 1

    rc = 0
    for d, n in sorted(by_dir.items(), key=lambda kv: -kv[1]):
        cohort = "auto_" + re.sub(r"[^A-Za-z0-9]+", "_", d.name)[:24]
        rc |= run([PY, "scripts/10_tiling.py", "--input", str(d), "--cohort", cohort, "--no-verify"])
    return rc


def extract_new(encoder: str, limit: int | None = None) -> int:
    """对缺特征的 patch 索引提特征。"""
    done = featured()
    idx = {p.stem: p for p in (ROOT / "data" / "patches").rglob("*.csv")}
    todo = {sid: p for sid, p in idx.items() if sid not in done and sid_of(sid) not in done}
    if not todo:
        print("没有需要提特征的切片")
        return 0
    if limit:
        todo = dict(list(todo.items())[:limit])
    print(f"需要提特征：{len(todo)} 张")

    by_dir: dict[Path, int] = {}
    for p in todo.values():
        by_dir[p.parent] = by_dir.get(p.parent, 0) + 1

    rc = 0
    for d, n in sorted(by_dir.items(), key=lambda kv: -kv[1]):
        cohort = d.name
        rc |= run([PY, "scripts/12_extract_features.py",
                   "--patches", str(d.relative_to(ROOT)),
                   "--cohort", cohort,
                   "--encoder", encoder, "--save", "--batch-size", "64",
                   "--wsi-dirs", "data/raw/tcga_ucec_normals",
                   "data/raw/tcga_ucec_tumor_subset", "TCGA-UCEC"])
    return rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="phikon-v2")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--wait", type=int, default=120, help="轮次间等待秒数")
    ap.add_argument("--only", choices=["patches", "features"], default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    for r in range(1, args.rounds + 1):
        print("=" * 70)
        print(f"第 {r}/{args.rounds} 轮  "
              f"本地切片 {len(local_slides())}  已分块 {len(patched())}  已提特征 {len(featured())}")
        print("=" * 70)
        if args.only != "features":
            tile_new(args.limit)
        if args.only != "patches":
            extract_new(args.encoder, args.limit)
        if r < args.rounds:
            print(f"\n等待 {args.wait}s 后进入下一轮…")
            time.sleep(args.wait)

    print("\n自动流水线结束")
    return 0


if __name__ == "__main__":
    sys.exit(main())
