#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
12_extract_features.py
======================
Stage 2：用冻结的病理基础模型提取 patch 特征，聚合为切片级特征包。

设计要点：
    - 只读 patch 坐标（Stage 1 产出），即时读取像素 → GPU 前向 → 写 HDF5；
    - 输出含 features / coords / patch_ids / slide_id / encoder_name，
      使后续 MIL 训练完全脱离 WSI 读取（迭代快）；
    - 支持多编码器；默认 Phikon-v2（开放权重，ViT-L/14，20x / 0.5 µm/px）。

输出：
    data/features/<encoder>/<slide_id>.h5
        features   float32 [N, D]
        coords     int32   [N, 2]   (level-0 坐标)
        patch_ids  str     [N]
        attrs: slide_id, encoder, mpp, patch_size, n_patches

用法：
    python scripts/12_extract_features.py --patches data/patches/tcga_normal --cohort tcga_normal --encoder phikon-v2 --limit 3
    python scripts/12_extract_features.py --patches data/patches/tcga_normal --cohort tcga_normal --encoder phikon-v2
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / ".pylibs"
if LIB.exists():
    sys.path.insert(0, str(LIB))

try:
    import openslide_bin  # noqa: F401
except ImportError:
    pass

# 已实测可用的编码器。维度与输入尺寸取自各自 config.json，不靠猜。
# 注意：phikon（v1）是 ViT-B（hidden 768），phikon-v2 是 ViT-L（hidden 1024）——
# 两者容量不同，作为"第二编码器对照"时须在论文中说明该差异。
ENCODERS = {
    "phikon-v2": {"path": ROOT / "models" / "phikon-v2", "dim": 1024, "input": 224, "norm": "imagenet"},
    "phikon": {"path": ROOT / "models" / "phikon", "dim": 768, "input": 224, "norm": "imagenet"},
    "vit-base-in": {"path": ROOT / "models" / "vit-base-in", "dim": 768, "input": 224, "norm": "imagenet"},
}

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_encoder(name: str, device: str):
    import torch
    from transformers import AutoModel
    spec = ENCODERS[name]
    if not (spec["path"] / "config.json").exists():
        raise FileNotFoundError(f"模型目录不完整：{spec['path']}（先跑 scripts/11_fetch_models.py）")
    model = AutoModel.from_pretrained(str(spec["path"]))
    model.eval().to(device)
    return model, spec


def preprocess(batch: np.ndarray, input_size: int) -> "np.ndarray":
    """uint8 [B,H,W,3] -> float32 [B,3,S,S]（ImageNet 归一化 + 最近邻缩放）。"""
    import torch
    import torch.nn.functional as F
    x = torch.from_numpy(batch).permute(0, 3, 1, 2).float().div_(255.0)
    if x.shape[-1] != input_size:
        x = F.interpolate(x, size=(input_size, input_size), mode="bilinear", align_corners=False)
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    return (x - mean) / std


def read_patches(slide, rows: list[dict], size: int) -> np.ndarray:
    batch = np.zeros((len(rows), size, size, 3), dtype=np.uint8)
    for i, r in enumerate(rows):
        try:
            region = slide.read_region((int(r["x_level0"]), int(r["y_level0"])),
                                       int(r["level"]), (size, size)).convert("RGB")
            arr = np.asarray(region)
            if arr.shape[0] == size and arr.shape[1] == size:
                batch[i] = arr
        except Exception:  # noqa: BLE001
            pass
    return batch


def extract_slide(wsi_path: Path, index_csv: Path, model, spec, device: str,
                  batch_size: int, out_path: Path | None) -> dict:
    import h5py
    import torch
    from openslide import OpenSlide

    rows = list(csv.DictReader(open(index_csv, encoding="utf-8-sig")))
    if not rows:
        return {"slide_id": wsi_path.stem, "n_patches": 0, "status": "empty_index"}

    slide = OpenSlide(str(wsi_path))
    feats: list[np.ndarray] = []
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            chunk = rows[start:start + batch_size]
            batch = read_patches(slide, chunk, int(chunk[0]["patch_size"]))
            x = preprocess(batch, spec["input"]).to(device, non_blocking=True)
            out = model(x)
            if hasattr(out, "last_hidden_state"):
                emb = out.last_hidden_state[:, 0]          # CLS token
            elif hasattr(out, "pooler_output") and out.pooler_output is not None:
                emb = out.pooler_output
            else:
                emb = out[0][:, 0]
            feats.append(emb.float().cpu().numpy())
    slide.close()

    features = np.concatenate(feats, axis=0).astype(np.float32)
    coords = np.array([[int(r["x_level0"]), int(r["y_level0"])] for r in rows], dtype=np.int32)
    patch_ids = np.array([r["patch_id"] for r in rows], dtype=object)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(out_path, "w") as fh:
            fh.create_dataset("features", data=features, compression="gzip", compression_opts=4)
            fh.create_dataset("coords", data=coords)
            dt = h5py.special_dtype(vlen=str)
            fh.create_dataset("patch_ids", data=patch_ids.astype(dt))
            fh.attrs["slide_id"] = wsi_path.stem
            fh.attrs["encoder"] = spec.get("name", "")
            fh.attrs["mpp"] = float(rows[0]["mpp"])
            fh.attrs["patch_size"] = int(rows[0]["patch_size"])
            fh.attrs["n_patches"] = len(rows)

    return {
        "slide_id": wsi_path.stem,
        "n_patches": len(rows),
        "feature_dim": int(features.shape[1]),
        "seconds": round(time.time() - t0, 1),
        "patches_per_sec": round(len(rows) / max(1e-6, time.time() - t0), 1),
        "status": "ok",
    }


def find_wsi(cohort_dirs: list[Path], slide_id: str) -> Path | None:
    for d in cohort_dirs:
        for ext in (".svs", ".tif", ".tiff", ".ndpi"):
            p = d / f"{slide_id}{ext}"
            if p.exists():
                return p
        hits = list(d.rglob(f"{slide_id}*"))
        for h in hits:
            if h.suffix.lower() in (".svs", ".tif", ".tiff", ".ndpi"):
                return h
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches", required=True, help="Stage 1 的 patch 索引目录")
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--encoder", default="phikon-v2", choices=list(ENCODERS))
    ap.add_argument("--wsi-dirs", nargs="*", default=None)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save", action="store_true", help="写出 HDF5（默认只统计）")
    ap.add_argument("--skip-existing", action="store_true",
                    help="跳过已存在 HDF5 的切片（用于中断后续跑，只补缺口）")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"编码器 {args.encoder}  设备 {device}  batch {args.batch_size}")

    model, spec = load_encoder(args.encoder, device)
    spec = dict(spec)
    spec["name"] = args.encoder

    idx_dir = Path(args.patches)
    if not idx_dir.is_absolute():
        idx_dir = ROOT / idx_dir
    indexes = sorted(idx_dir.glob("*.csv"))
    if args.limit:
        indexes = indexes[: args.limit]

    wsi_dirs = [Path(d) if Path(d).is_absolute() else ROOT / d
                for d in (args.wsi_dirs or ["data/raw/tcga_ucec_normals", "data/raw/tcga_ucec_tumor_subset", "TCGA-UCEC"])]

    out_dir = ROOT / "data" / "features" / args.encoder / args.cohort
    qc: list[dict] = []
    n_skipped = 0
    for i, idx in enumerate(indexes, 1):
        slide_id = idx.stem
        if args.skip_existing and (out_dir / f"{slide_id}.h5").exists():
            n_skipped += 1
            continue
        wsi = find_wsi(wsi_dirs, slide_id)
        if wsi is None:
            print(f"[{i}/{len(indexes)}] {slide_id[:44]:46s} 未找到对应 WSI")
            qc.append({"slide_id": slide_id, "status": "wsi_not_found"})
            continue
        out_path = out_dir / f"{slide_id}.h5" if args.save else None
        try:
            rec = extract_slide(wsi, idx, model, spec, device, args.batch_size, out_path)
            rec["wsi"] = str(wsi)
            qc.append(rec)
            print(f"[{i}/{len(indexes)}] {slide_id[:44]:46s} patches={rec['n_patches']:6d} "
                  f"dim={rec.get('feature_dim','-')} {rec.get('seconds',0):6.1f}s "
                  f"{rec.get('patches_per_sec',0):7.1f} p/s {rec['status']}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[{i}/{len(indexes)}] {slide_id[:44]:46s} 失败 {type(exc).__name__}: {exc}", flush=True)
            qc.append({"slide_id": slide_id, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})

    qc_path = ROOT / "data" / "qc" / f"feature_extraction_{args.encoder}_{args.cohort}.csv"
    qc_path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in qc for k in r})
    with open(qc_path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(qc)

    ok = [r for r in qc if r.get("status") == "ok"]
    total = sum(r.get("n_patches", 0) for r in ok)
    if args.skip_existing:
        print(f"（跳过已存在 {n_skipped} 张）")
    print(f"\n完成：{len(ok)}/{len(qc)} 张，共 {total} 个 patch 特征")
    print(f"QC -> {qc_path}")
    if args.save:
        print(f"特征 -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
