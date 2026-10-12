#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
54_tile_biopsy.py
=================
针对**组织占比极低**的活检 WSI（如 ENDO-AID Pipelle 活检，组织仅 0.16–1.63%）
做专门的分块：放宽筛选条件，保证每张切片仍能取到足够 patch。

为什么需要专用脚本：
    10_tiling.py 的默认阈值（tissue_thresh=0.5，且低倍粗筛要求缩略图区域组织占比
    ≥ 阈值的 0.6 倍）是为组织量充足的切除标本设计的。活检组织量少、碎片化，
    用默认阈值会把整张切片全部过滤掉（已实测：ENDO-AID 组织占比仅 0.16–1.63%）。

策略：
    1. 降低 patch 级组织阈值（默认 0.25）；
    2. 低倍粗筛阈值进一步放宽，并设每张切片的最小 patch 数下限；
    3. 若仍不足，改为「按组织像素优先采样」而不是全图网格扫描。

用法：
    python scripts/54_tile_biopsy.py --input data/raw/endoaid_external/extracted --cohort endoaid_biopsy
    python scripts/54_tile_biopsy.py --input ... --cohort endoaid_biopsy --min-patches 500
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

try:
    import openslide_bin  # noqa: F401
except ImportError:
    pass
from openslide import OpenSlide  # noqa: E402

from importlib import import_module  # noqa: E402
til = import_module("10_tiling")


def tile_low_tissue(path: Path, out_dir: Path, patch: int, tissue_thresh: float,
                    coarse_ratio: float, min_patches: int, max_patches: int | None,
                    target_mpp: float) -> tuple[list[dict], dict]:
    slide = OpenSlide(str(path))
    slide_id = til.canonical_slide_id(path)
    t0 = time.time()
    level, actual_mpp = til.pick_level(slide, target_mpp)

    mask, tw, th = til.tissue_mask_from_thumbnail(slide)
    mask = til.refine_mask(mask)
    tissue_pct = float(mask.mean() * 100)

    W, H = slide.level_dimensions[level]
    ds = slide.level_downsamples[level]
    sx, sy = W / tw, H / th
    step = patch

    coords = np.argwhere(mask)
    seen = set()
    cand: list[tuple[float, int, int]] = []
    for (ty, tx) in coords:
        gx = (int(tx * sx) // step) * step
        gy = (int(ty * sy) // step) * step
        if gx + step > W or gy + step > H or (gx, gy) in seen:
            continue
        seen.add((gx, gy))
        tx0, ty0 = int(gx / sx), int(gy / sy)
        tx1, ty1 = int((gx + step) / sx) + 1, int((gy + step) / sy) + 1
        sub = mask[max(0, ty0):min(th, ty1), max(0, tx0):min(tw, tx1)]
        if sub.size == 0:
            continue
        frac = float(sub.mean())
        if frac < coarse_ratio:
            continue
        cand.append((frac, gx, gy))

    # 优先取组织比例高的块
    cand.sort(key=lambda c: -c[0])
    rows: list[dict] = []
    for frac, gx, gy in cand:
        lx0, ly0 = int(gx * ds), int(gy * ds)
        real = til.patch_tissue_fraction(slide, lx0, ly0, level, patch)
        if real < tissue_thresh:
            continue
        rows.append({
            "slide_id": slide_id,
            "patch_id": f"{slide_id}_x{gx}_y{gy}",
            "x_level0": lx0, "y_level0": ly0,
            "x_level": gx, "y_level": gy,
            "level": level, "patch_size": patch,
            "mpp": round(actual_mpp, 4),
            "tissue_fraction": round(real, 4),
        })
        if max_patches and len(rows) >= max_patches:
            break
        if len(rows) >= min_patches and not max_patches:
            # 达到下限后再多取一些以提升代表性，但设上限避免爆炸
            if len(rows) >= min_patches * 3:
                break

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{slide_id}.csv", "w", encoding="utf-8-sig", newline="") as fh:
        fields = list(rows[0].keys()) if rows else [
            "slide_id", "patch_id", "x_level0", "y_level0", "x_level", "y_level",
            "level", "patch_size", "mpp", "tissue_fraction"]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    qc = {"slide_id": slide_id, "file": path.name, "width": slide.dimensions[0],
          "height": slide.dimensions[1], "level_used": level,
          "mpp_used": round(actual_mpp, 4), "tissue_pct_thumb": round(tissue_pct, 2),
          "n_candidates": len(cand), "n_patches": len(rows),
          "patch_size": patch, "tissue_thresh": tissue_thresh,
          "seconds": round(time.time() - t0, 1),
          "status": "ok" if rows else "no_tissue"}
    slide.close()
    return rows, qc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--mpp", type=float, default=0.5)
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--tissue-thresh", type=float, default=0.25,
                    help="patch 级组织阈值（活检用 0.25，低于切除标本的 0.5）")
    ap.add_argument("--coarse-ratio", type=float, default=0.10,
                    help="低倍粗筛阈值（缩略图区域组织占比）")
    ap.add_argument("--min-patches", type=int, default=400)
    ap.add_argument("--max-patches", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    src = Path(args.input)
    if not src.is_absolute():
        src = ROOT / src
    files: list[Path] = []
    for ext in ("*.tif", "*.tiff", "*.svs", "*.ndpi"):
        files.extend(sorted(src.rglob(ext)))
    files = [f for f in files if not f.name.startswith(".")]
    if args.limit:
        files = files[: args.limit]

    out_dir = ROOT / "data" / "patches" / args.cohort
    qc_path = ROOT / "data" / "qc" / f"tiling_qc_{args.cohort}.csv"
    qc_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"队列 {args.cohort}：{len(files)} 张活检切片")
    print(f"参数：mpp={args.mpp} patch={args.patch} tissue_thresh={args.tissue_thresh} "
          f"coarse_ratio={args.coarse_ratio} min_patches={args.min_patches}")

    all_qc: list[dict] = []
    for i, f in enumerate(files, 1):
        try:
            rows, qc = tile_low_tissue(f, out_dir, args.patch, args.tissue_thresh,
                                       args.coarse_ratio, args.min_patches,
                                       args.max_patches, args.mpp)
            all_qc.append(qc)
            print(f"[{i}/{len(files)}] {qc['slide_id'][:42]:44s} tissue={qc['tissue_pct_thumb']:5.2f}%  "
                  f"cand={qc['n_candidates']:6d}  patches={qc['n_patches']:5d}  "
                  f"{qc['seconds']:6.1f}s  {qc['status']}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[{i}/{len(files)}] {f.name}: 失败 {type(exc).__name__}: {exc}", flush=True)
            all_qc.append({"slide_id": f.stem, "file": f.name, "status": "failed",
                           "error": f"{type(exc).__name__}: {exc}"})

    fields = sorted({k for r in all_qc for k in r})
    with open(qc_path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(all_qc)

    ok = [r for r in all_qc if r.get("status") == "ok"]
    total = sum(r.get("n_patches", 0) for r in ok)
    print(f"\n完成：成功 {len(ok)}/{len(all_qc)}，共 {total} 个 patch")
    if ok:
        cnts = [r["n_patches"] for r in ok]
        print(f"每切片 patch 数：min={min(cnts)} 中位={int(np.median(cnts))} max={max(cnts)}")
    print(f"QC -> {qc_path}")
    print(f"patch 索引 -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
