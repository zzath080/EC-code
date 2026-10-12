#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
04_verify_wsi_reader.py
=======================
验证 WSI 读取链路：对本地已有切片做真实读取测试，输出可用于论文的 QC 表。

支持优先级：
    1. openslide (openslide-python + openslide-bin)
    2. tiffslide (纯 Python, 基于 tifffile)

输出：
    data/qc/wsi_read_test.csv      每张切片的分辨率/MPP/倍率/组织占比
    data/qc/reader_report.json     读取器可用性报告
"""

from __future__ import annotations

import csv
import json
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
QC = ROOT / "data" / "qc"


def open_slide(path: Path):
    """返回 (reader_name, slide_object)。"""
    errors = []
    try:
        from openslide import OpenSlide
        return "openslide", OpenSlide(str(path))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"openslide: {type(exc).__name__}: {exc}")
    try:
        from tiffslide import TiffSlide
        return "tiffslide", TiffSlide(str(path))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"tiffslide: {type(exc).__name__}: {exc}")
    raise RuntimeError("no WSI reader available: " + " | ".join(errors))


def inspect(path: Path) -> dict:
    row = {"file": path.name, "path": str(path.relative_to(ROOT)), "status": "ok"}
    try:
        reader, slide = open_slide(path)
        row["reader"] = reader
        props = dict(slide.properties)
        row["width"] = slide.dimensions[0]
        row["height"] = slide.dimensions[1]
        row["level_count"] = slide.level_count
        row["level_dims"] = "|".join(f"{w}x{h}" for w, h in slide.level_dimensions)
        row["downsamples"] = "|".join(f"{d:.1f}" for d in slide.level_downsamples)
        mpp_x = props.get("openslide.mpp-x") or props.get("tiff.XResolution")
        mpp_y = props.get("openslide.mpp-y") or props.get("tiff.YResolution")
        row["mpp_x"] = mpp_x
        row["mpp_y"] = mpp_y
        row["magnification"] = props.get("openslide.objective-power", "")
        row["vendor"] = props.get("openslide.vendor", "")
        row["slide_format"] = props.get("openslide.region-count", "")

        # 缩略图 + 组织占比（灰度阈值法，Otsu 近似用固定 220）
        thumb = slide.get_thumbnail((1024, 1024)).convert("RGB")
        arr = np.asarray(thumb.convert("L"))
        tissue_pct = float((arr < 220).mean() * 100)
        row["thumb_w"], row["thumb_h"] = thumb.size
        row["tissue_pct"] = round(tissue_pct, 2)

        # 真实读取一个 patch，验证像素级可读
        level = min(1, slide.level_count - 1)
        size = 256
        w, h = slide.level_dimensions[level]
        if w > size and h > size:
            patch = slide.read_region((max(0, w // 2 - size // 2), max(0, h // 2 - size // 2)), level, (size, size))
            parr = np.asarray(patch.convert("RGB"))
            row["patch_readable"] = True
            row["patch_mean_rgb"] = "|".join(f"{v:.1f}" for v in parr.reshape(-1, 3).mean(axis=0))
        else:
            row["patch_readable"] = False
        slide.close()
    except Exception as exc:  # noqa: BLE001
        row["status"] = "failed"
        row["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    return row


def main() -> int:
    QC.mkdir(parents=True, exist_ok=True)
    candidates: list[Path] = []
    for d in [ROOT / "TCGA-UCEC", ROOT / "data" / "raw"]:
        if d.exists():
            for ext in ("*.svs", "*.tif", "*.tiff"):
                candidates.extend(d.rglob(ext))
    candidates = sorted(set(candidates))
    print(f"待测切片 {len(candidates)} 张")

    rows = [inspect(p) for p in candidates]
    if not rows:
        print("[WARN] 未找到本地切片")
        return 1

    fields = sorted({k for r in rows for k in r})
    with open(QC / "wsi_read_test.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r["status"] == "ok"]
    report = {
        "tested": len(rows),
        "ok": len(ok),
        "failed": len(rows) - len(ok),
        "readers_used": sorted({r.get("reader", "") for r in ok}),
    }
    (QC / "reader_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))
    for r in ok[:5]:
        print(f"  {r['file'][:45]:47s} {r['width']}x{r['height']}  mpp={r.get('mpp_x')}  tissue={r.get('tissue_pct')}%  reader={r['reader']}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
