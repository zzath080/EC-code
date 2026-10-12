#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
10_tiling.py
============
Stage 1：组织分割 + 分块索引（不保存 patch 图像，只保存坐标）。

设计要点（写论文方法学时可直接引用）：
    - 组织分割在低倍缩略图上做（灰度 + Otsu），再形态学去噪，避免逐块读像素；
    - 分块在目标倍率下做：默认 0.5 µm/px（≈20x），与 Phikon-v2 / UNI2-h 的预训练一致；
    - patch 尺寸 256×256（Phikon-v2 内部再 resize 到 224）；
    - 组织比例阈值默认 0.5，低倍快速筛 + 高倍精确复核；
    - **只保存坐标与元数据**，patch 图像在特征提取阶段即时读取，省磁盘。

输出：
    data/patches/<cohort>/<slide_id>.csv        每张切片的 patch 索引
    data/qc/tiling_qc.csv                       每张切片的汇总 QC

用法：
    python scripts/10_tiling.py --input data/raw/tcga_ucec_normals --cohort tcga_normal
    python scripts/10_tiling.py --input TCGA-UCEC --cohort tcga_tumor --mpp 0.5 --patch 256
    python scripts/10_tiling.py --input data/raw/tcga_ucec_normals --cohort tcga_normal --limit 3 --dry-run
"""

from __future__ import annotations

import argparse
import csv
import re
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


# ---------------------------------------------------------------- 图像工具
def otsu_threshold(gray: np.ndarray) -> int:
    """经典 Otsu 阈值；纯 numpy 实现，避免额外依赖。"""
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 128
    omega = np.cumsum(hist) / total
    mu = np.cumsum(hist * np.arange(256)) / total
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    denom[denom == 0] = 1e-12
    sigma_b = (mu_t * omega - mu) ** 2 / denom
    return int(np.argmax(sigma_b))


def tissue_mask_from_thumbnail(slide: OpenSlide, thumb_max: int = 2048,
                               sat_thresh: int = 20, dark_thresh: int = 220) -> tuple[np.ndarray, int, int]:
    """在缩略图上生成组织掩膜。返回 (mask, thumb_w, thumb_h)。"""
    thumb = slide.get_thumbnail((thumb_max, thumb_max)).convert("RGB")
    arr = np.asarray(thumb)
    gray = np.asarray(thumb.convert("L"))
    hsv_max = np.max(arr, axis=2)
    hsv_min = np.min(arr, axis=2)
    saturation = hsv_max - hsv_min
    # 组织 = 不太亮 或 有明显染色饱和度
    mask = (gray < dark_thresh) | (saturation > sat_thresh)
    # 去掉纯白背景残留
    mask &= ~((gray > 230) & (saturation < 10))
    return mask, thumb.size[0], thumb.size[1]


def refine_mask(mask: np.ndarray) -> np.ndarray:
    """形态学开运算去噪（纯 numpy，用 3x3 邻域投票近似）。"""
    m = mask.astype(np.uint8)
    # erode
    e = m.copy()
    e[1:, :] &= m[:-1, :]
    e[:-1, :] &= m[1:, :]
    e[:, 1:] &= m[:, :-1]
    e[:, :-1] &= m[:, 1:]
    # dilate（两次）
    for _ in range(2):
        d = e.copy()
        d[1:, :] |= e[:-1, :]
        d[:-1, :] |= e[1:, :]
        d[:, 1:] |= e[:, :-1]
        d[:, :-1] |= e[:, 1:]
        e = d
    return e.astype(bool)


def patch_tissue_fraction(slide: OpenSlide, x: int, y: int, level: int, size: int,
                          sat_thresh: int = 20, dark_thresh: int = 220) -> float:
    """读取单块并计算组织比例（用于高倍精确复核）。"""
    try:
        region = slide.read_region((x, y), level, (size, size)).convert("RGB")
    except Exception:  # noqa: BLE001
        return 0.0
    arr = np.asarray(region)
    if arr.shape[0] != size or arr.shape[1] != size:
        return 0.0
    gray = arr.mean(axis=2)
    sat = arr.max(axis=2) - arr.min(axis=2)
    m = (gray < dark_thresh) | (sat > sat_thresh)
    m &= ~((gray > 230) & (sat < 10))
    return float(m.mean())


# ---------------------------------------------------------------- 主流程
def pick_level(slide: OpenSlide, target_mpp: float) -> tuple[int, float]:
    """挑选最接近目标 µm/px 的层级；返回 (level, 该层实际 mpp)。"""
    mpp_x = slide.properties.get("openslide.mpp-x")
    if mpp_x is None:
        return 0, 0.0
    base = float(mpp_x)
    best, best_err = 0, float("inf")
    for lv, ds in enumerate(slide.level_downsamples):
        err = abs(base * ds - target_mpp)
        if err < best_err:
            best, best_err = lv, err
    return best, base * slide.level_downsamples[best]


def canonical_slide_id(path: Path) -> str:
    """统一的 slide_id：去掉扩展名与 GDC 文件名里的 UUID 段。

    必要性：GDC 原始文件名形如 TCGA-2E-A9G8-01Z-00-DX1.<UUID>.svs，而公开特征库
    与划分清单用不带 UUID 的 TCGA-2E-A9G8-01Z-00-DX1。若不统一，同一张切片会在
    patch 索引 / 特征 / 划分清单之间产生 ID 不一致，导致样本被静默丢弃。
    """
    stem = re.sub(r"\.(svs|tif|tiff|ndpi)$", "", path.name, flags=re.I)
    return re.sub(r"\.[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$", "", stem)


def tile_slide(path: Path, out_dir: Path, target_mpp: float, patch: int,
               tissue_thresh: float, max_patches: int | None,
               verify_high_mag: bool) -> tuple[list[dict], dict]:
    slide = OpenSlide(str(path))
    slide_id = canonical_slide_id(path)
    t0 = time.time()
    mpp_x = slide.properties.get("openslide.mpp-x")
    level, actual_mpp = pick_level(slide, target_mpp)

    mask, tw, th = tissue_mask_from_thumbnail(slide)
    mask = refine_mask(mask)
    tissue_pct = float(mask.mean() * 100)

    level_dims = slide.level_dimensions
    level_ds = slide.level_downsamples
    W, H = level_dims[level]
    # 缩略图坐标 -> 目标层坐标
    sx, sy = W / tw, H / th
    step = patch

    rows: list[dict] = []
    coords = np.argwhere(mask)  # (y, x) in thumbnail space
    seen = set()
    for (ty, tx) in coords:
        lx = int(tx * sx)
        ly = int(ty * sy)
        # 对齐到 patch 网格，避免同一块被重复计数
        gx = (lx // step) * step
        gy = (ly // step) * step
        if gx + step > W or gy + step > H:
            continue
        key = (gx, gy)
        if key in seen:
            continue
        seen.add(key)

        # 低倍快速筛选：该 patch 覆盖的缩略图区域有多少是组织
        tx0, ty0 = int(gx / sx), int(gy / sy)
        tx1, ty1 = int((gx + step) / sx) + 1, int((gy + step) / sy) + 1
        sub = mask[max(0, ty0):min(th, ty1), max(0, tx0):min(tw, tx1)]
        if sub.size == 0 or sub.mean() < tissue_thresh * 0.6:
            continue

        # 高倍精确复核
        frac = None
        if verify_high_mag:
            lx0 = int(gx * level_ds[level])
            ly0 = int(gy * level_ds[level])
            frac = patch_tissue_fraction(slide, lx0, ly0, level, patch)
            if frac < tissue_thresh:
                continue

        rows.append({
            "slide_id": slide_id,
            "patch_id": f"{slide_id}_x{gx}_y{gy}",
            "x_level0": int(gx * level_ds[level]),
            "y_level0": int(gy * level_ds[level]),
            "x_level": gx,
            "y_level": gy,
            "level": level,
            "patch_size": patch,
            "mpp": round(actual_mpp, 4),
            "tissue_fraction": round(float(frac if frac is not None else sub.mean()), 4),
        })
        if max_patches and len(rows) >= max_patches:
            break

    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{slide_id}.csv"
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else
                           ["slide_id", "patch_id", "x_level0", "y_level0", "x_level",
                            "y_level", "level", "patch_size", "mpp", "tissue_fraction"])
        w.writeheader()
        w.writerows(rows)

    qc = {
        "slide_id": slide_id,
        "file": path.name,
        "width": slide.dimensions[0],
        "height": slide.dimensions[1],
        "level_used": level,
        "mpp_base": mpp_x,
        "mpp_used": round(actual_mpp, 4),
        "level_count": slide.level_count,
        "tissue_pct_thumb": round(tissue_pct, 2),
        "n_patches": len(rows),
        "patch_size": patch,
        "tissue_thresh": tissue_thresh,
        "seconds": round(time.time() - t0, 1),
        "status": "ok" if rows else "no_tissue",
    }
    slide.close()
    return rows, qc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="切片文件或目录")
    ap.add_argument("--cohort", required=True, help="队列名，用于输出目录")
    ap.add_argument("--mpp", type=float, default=0.5, help="目标分辨率 µm/px（0.5≈20x）")
    ap.add_argument("--patch", type=int, default=256)
    ap.add_argument("--tissue-thresh", type=float, default=0.5)
    ap.add_argument("--max-patches", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 张（试跑）")
    ap.add_argument("--no-verify", action="store_true", help="跳过高倍精确复核（更快）")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = Path(args.input)
    if not src.is_absolute():
        src = ROOT / src
    files: list[Path] = []
    if src.is_dir():
        for ext in ("*.svs", "*.tif", "*.tiff", "*.ndpi"):
            files.extend(sorted(src.rglob(ext)))
    elif src.is_file():
        files.append(src)
    if args.limit:
        files = files[: args.limit]

    out_dir = ROOT / "data" / "patches" / args.cohort
    qc_path = ROOT / "data" / "qc" / f"tiling_qc_{args.cohort}.csv"
    qc_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"队列 {args.cohort}：{len(files)} 张切片，目标 {args.mpp} µm/px，patch {args.patch}")
    if args.dry_run:
        for f in files[:10]:
            print("  ", f.name)
        return 0

    all_qc: list[dict] = []
    for i, f in enumerate(files, 1):
        try:
            rows, qc = tile_slide(f, out_dir, args.mpp, args.patch, args.tissue_thresh,
                                  args.max_patches, not args.no_verify)
            all_qc.append(qc)
            print(f"[{i}/{len(files)}] {qc['slide_id'][:44]:46s} "
                  f"tissue={qc['tissue_pct_thumb']:5.1f}%  patches={qc['n_patches']:6d}  "
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
    total_patches = sum(r.get("n_patches", 0) for r in all_qc)
    print(f"\n完成：成功 {len(ok)}/{len(all_qc)}，共 {total_patches} 个 patch")
    print(f"QC -> {qc_path}")
    print(f"patch 索引 -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
