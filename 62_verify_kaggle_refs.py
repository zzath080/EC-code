#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
62_verify_kaggle_refs.py
========================
批量核实 Kaggle 数据集 ref 是否有效（403 表示不存在或已删除），
并给出体量与许可，供 61_build_source_matrix.py 挑选。

用法：python scripts/62_verify_kaggle_refs.py
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (endo-wsi-research)"}

REFS = [
    # 多器官 / 不同器官
    "andrewmvd/lung-and-colon-cancer-histopathological-images",
    "apngupta/colon-cancer-datasets",
    "kmader/kidney-histopathology",
    "truthisneverlinear/brecahad",
    "ashery/breast-cancer-histopathology",
    "paultimothymooney/breast-histopathology-images",
    "ambarish/breakhis",
    "alovestocode/breast-histopathology-images-modified",
    # 其他器官
    "tahirlee/uterine-leiomyosarcoma-histopathology",
    "samihamuntahamahin/uterus-cancer",
    "samihamuntahamahin/endometrial-histopathology",
    "bitsnpieces/ovarian-cancer-and-subtypes-dataset-histopathology",
    "obulisainaren/endometrial-cancer",
    "harsha5474/oral-cancer-dataset",
    "somnathsharma921/lymphoma-histopathology",
    "mohamedmustafa/monuseg-segmentation-of-nuclei-in-cells",
]


def view(ref: str) -> dict:
    url = f"https://www.kaggle.com/api/v1/datasets/view/{ref}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return json.load(r)


def main() -> int:
    ok, fail = [], []
    for ref in REFS:
        try:
            d = view(ref)
            b = d.get("totalBytes") or d.get("totalBytesNullable") or 0
            lic = d.get("licenseNameNullable") or d.get("licenseName")
            title = d.get("titleNullable") or d.get("title")
            ok.append({"ref": ref, "bytes": b, "license": lic, "title": title})
            print(f"  OK   {ref:66s} {b/1024**2:8.0f} MB  {str(lic)[:22]}")
        except urllib.error.HTTPError as exc:
            fail.append({"ref": ref, "status": exc.code})
            print(f"  HTTP {exc.code} {ref}")
        except Exception as exc:  # noqa: BLE001
            fail.append({"ref": ref, "error": type(exc).__name__})
            print(f"  ERR  {ref}: {type(exc).__name__}")
    print(f"\n可用 {len(ok)} / {len(REFS)}")
    (__import__("pathlib").Path(__file__).resolve().parents[1] /
     "data" / "manifests" / "kaggle_ref_check.json").write_text(
        json.dumps({"ok": ok, "fail": fail}, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
