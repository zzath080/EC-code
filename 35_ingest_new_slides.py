#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
35_ingest_new_slides.py
=======================
把新下载的切片接入流水线：自动分块 + 提特征，并纳入统一的特征目录与清单。

相比 30_auto_pipeline.py，本脚本的差异：
    - 明确只处理**指定目录**（如 data/raw/tcga_ucec_01z），避免误扫
    - 分块后自动提特征，一步到位
    - 结束后重建训练清单并打印跨标本类型可用的样本量

用法：
    python scripts/35_ingest_new_slides.py --dir data/raw/tcga_ucec_01z --cohort tcga_01z
    python scripts/35_ingest_new_slides.py --dir data/raw/tcga_ucec_01z --cohort tcga_01z --limit 6
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
WSI_EXTS = ("*.svs", "*.tif", "*.tiff", "*.ndpi")


def sid_of(name: str) -> str:
    s = re.sub(r"\.(svs|h5|tif|tiff|ndpi)$", "", name, flags=re.I)
    return re.sub(r"\.[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$", "", s)


def run(cmd: list[str]) -> int:
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    return subprocess.run(cmd, cwd=str(ROOT)).returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--encoder", default="phikon-v2")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-tiling", action="store_true")
    ap.add_argument("--skip-features", action="store_true")
    args = ap.parse_args()

    src = Path(args.dir)
    if not src.is_absolute():
        src = ROOT / src
    files = []
    for ext in WSI_EXTS:
        files.extend(sorted(src.rglob(ext)))
    files = [f for f in files if not f.name.startswith(".")]
    print(f"目录 {src}：{len(files)} 张切片")
    if not files:
        print("[WARN] 没有切片可处理")
        return 1

    patch_dir = ROOT / "data" / "patches" / args.cohort
    have = {p.stem for p in patch_dir.glob("*.csv")} if patch_dir.exists() else set()
    need_tile = [f for f in files if sid_of(f.name) not in have and f.name.rsplit(".", 1)[0] not in have]
    if args.limit:
        need_tile = need_tile[: args.limit]
    print(f"需要分块：{len(need_tile)} 张（已有索引 {len(have)} 张）")

    if not args.skip_tiling and need_tile:
        # 只处理需要的切片：临时目录放白名单，避免重复处理
        rc = run([PY, "scripts/10_tiling.py", "--input", str(src), "--cohort", args.cohort, "--no-verify"])
        if rc != 0:
            print(f"[WARN] 分块返回 {rc}")

    feat_dir = ROOT / "data" / "features" / args.encoder / args.cohort
    have_f = {p.stem for p in feat_dir.glob("*.h5")} if feat_dir.exists() else set()
    idx = {p.stem for p in patch_dir.glob("*.csv")} if patch_dir.exists() else set()
    need_feat = idx - have_f
    print(f"需要提特征：{len(need_feat)} 张")

    if not args.skip_features and need_feat:
        rc = run([PY, "scripts/12_extract_features.py",
                  "--patches", str(patch_dir.relative_to(ROOT)),
                  "--cohort", args.cohort,
                  "--encoder", args.encoder, "--save", "--batch-size", "64",
                  "--wsi-dirs", str(src.relative_to(ROOT)) if src.is_relative_to(ROOT) else str(src),
                  "data/raw/tcga_ucec_normals", "data/raw/tcga_ucec_tumor_subset", "TCGA-UCEC"])
        if rc != 0:
            print(f"[WARN] 特征提取返回 {rc}")

    # 重建清单并报告可用样本量
    run([PY, "scripts/20_merge_feature_manifest.py", "--cohort", "tcga_pair"])
    man = ROOT / "data" / "manifests" / "tcga_pair_train_manifest.csv"
    if man.exists():
        rows = list(csv.DictReader(open(man, encoding="utf-8-sig")))
        by_kind: dict[str, dict[str, int]] = {}
        for r in rows:
            parts = r["slide_id"].split("-")
            code = parts[3] if len(parts) > 3 else "?"
            kind = "diag" if code.endswith("Z") else "tissue"
            d = by_kind.setdefault(kind, {"n": 0, "pos": 0, "neg": 0})
            d["n"] += 1
            d["pos" if r["label"] == "1" else "neg"] += 1
        print("\n跨标本类型可用样本量：")
        for k, v in by_kind.items():
            print(f"  {k:8s} 共 {v['n']:3d} 张（恶性 {v['pos']:3d} / 良性 {v['neg']:3d}）")
        d = by_kind.get("diag", {})
        if d.get("pos", 0) >= 5 and d.get("neg", 0) >= 5:
            print("  → 目标域正负样本均已 ≥5，跨域 AUC 可估计 ✅")
        else:
            print("  → 目标域样本仍不足，AUC 尚不可估计")
    return 0


if __name__ == "__main__":
    sys.exit(main())
