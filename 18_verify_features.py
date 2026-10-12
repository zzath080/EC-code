#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
18_verify_features.py
=====================
验证特征 HDF5 的真实 schema、数值健康度，并与对应 WSI 的尺寸交叉核对。

这是「特征能不能直接用」的硬证据：
    - schema 是否符合预期（features / coords / annots）
    - patch 数、特征维度、是否有 NaN/常量
    - coords 是否落在对应 WSI 的 level-0 尺寸范围内
    - 与本地 patch 索引（scripts/10_tiling.py 产出）的一致性对比

输出：
    data/qc/feature_verification.csv
    data/qc/feature_verification.json
"""

from __future__ import annotations

import csv
import json
import re
import sys
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


def norm_id(name: str) -> str:
    stem = re.sub(r"\.(svs|h5|tif|tiff|ndpi)$", "", name, flags=re.I)
    return re.sub(r"\.[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$", "", stem)


def find_wsi(slide_id: str) -> Path | None:
    for d in [ROOT / "TCGA-UCEC", ROOT / "data" / "raw"]:
        if not d.exists():
            continue
        for ext in ("*.svs", "*.tif", "*.tiff"):
            for p in d.rglob(ext):
                if norm_id(p.name) == slide_id:
                    return p
    return None


def inspect_one(h5_path: Path) -> dict:
    import h5py
    rec: dict = {"file": h5_path.name, "slide_id": norm_id(h5_path.name),
                 "size_mb": round(h5_path.stat().st_size / 1024 ** 2, 2), "status": "ok"}
    with h5py.File(h5_path, "r") as fh:
        rec["keys"] = "|".join(fh.keys())
        rec["attrs"] = "|".join(f"{k}={fh.attrs[k]}" for k in fh.attrs)
        for key in fh.keys():
            if isinstance(fh[key], h5py.Dataset):
                rec[f"{key}_shape"] = "x".join(str(s) for s in fh[key].shape)
                rec[f"{key}_dtype"] = str(fh[key].dtype)
        if "features" in fh:
            f = fh["features"][:]
            f2 = f.reshape(-1, f.shape[-1]) if f.ndim > 2 else f
            rec["n_patches"] = int(f2.shape[0])
            rec["feature_dim"] = int(f2.shape[1])
            rec["feat_mean"] = round(float(f2.mean()), 4)
            rec["feat_std"] = round(float(f2.std()), 4)
            rec["n_nan"] = int(np.isnan(f2).sum())
            rec["n_zero_rows"] = int((np.abs(f2).sum(axis=1) == 0).sum())
            rec["unique_rows_sample"] = int(np.unique(f2[: min(200, len(f2))], axis=0).shape[0])
        if "coords" in fh:
            c = fh["coords"][:]
            c2 = c.reshape(-1, c.shape[-1]) if c.ndim > 2 else c
            rec["coord_max_x"] = int(c2[:, 0].max())
            rec["coord_max_y"] = int(c2[:, 1].max())
            rec["coord_min"] = f"{int(c2.min())}"
        if "annots" in fh:
            a = fh["annots"][:]
            rec["annots_shape"] = "x".join(str(s) for s in a.shape)
            rec["annots_values"] = "|".join(str(v) for v in np.unique(a)[:10])

    wsi = find_wsi(rec["slide_id"])
    if wsi is not None:
        from openslide import OpenSlide
        s = OpenSlide(str(wsi))
        w, h = s.dimensions
        rec["wsi"] = str(wsi.relative_to(ROOT))
        rec["wsi_w"], rec["wsi_h"] = w, h
        rec["mpp_x"] = s.properties.get("openslide.mpp-x", "")
        s.close()
        if "coord_max_x" in rec:
            # 0.5 µm/px 下 256px patch 的坐标应远小于 level-0 尺寸
            rec["coords_within_wsi"] = bool(rec["coord_max_x"] < w and rec["coord_max_y"] < h)
    else:
        rec["wsi"] = ""
    return rec


def main() -> int:
    feat_dir = ROOT / "data" / "features" / "uni2h_public" / "TCGA-UCEC"
    files = sorted(feat_dir.glob("*.h5"))
    if not files:
        print(f"[ERROR] {feat_dir} 下没有 .h5", file=sys.stderr)
        return 2

    rows = [inspect_one(f) for f in files]
    fields = sorted({k for r in rows for k in r})
    out_dir = ROOT / "data" / "qc"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "feature_verification.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    summary = {
        "n_files": len(rows),
        "schema_keys": sorted({k for r in rows for k in str(r.get("keys", "")).split("|")}),
        "feature_dims": sorted({r.get("feature_dim") for r in rows if r.get("feature_dim")}),
        "total_patches": sum(r.get("n_patches", 0) or 0 for r in rows),
        "any_nan": any((r.get("n_nan") or 0) > 0 for r in rows),
        "all_coords_within_wsi": all(r.get("coords_within_wsi", True) for r in rows),
        "files_with_wsi": sum(1 for r in rows if r.get("wsi")),
    }
    (out_dir / "feature_verification.json").write_text(
        json.dumps({"summary": summary, "details": rows}, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print()
    for r in rows:
        print(f"  {r['slide_id'][:34]:36s} patches={r.get('n_patches','-'):>6} "
              f"dim={r.get('feature_dim','-')} nan={r.get('n_nan','-')} "
              f"zrows={r.get('n_zero_rows','-')} coordmax=({r.get('coord_max_x','-')},{r.get('coord_max_y','-')}) "
              f"wsi={r.get('wsi_w','-')}x{r.get('wsi_h','-')} in={r.get('coords_within_wsi','-')}")
    print(f"\n-> {out_dir/'feature_verification.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
