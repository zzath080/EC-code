#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
61_build_source_matrix.py
=========================
【路线 1 主线】把跨来源泛化矩阵的数据源从 4 个扩到 6–8 个。

与 57_harvest_patch_datasets.py 的区别：
    57 是"广撒网找候选"；本脚本是**定向建设**——只收「能明确二分为恶性/良性」的
    组织病理 patch 数据集，并直接给出每个来源的类别映射，避免事后猜测。

设计原则：
    - 优先**不同器官**（乳腺/结肠/肺/口腔/肾/淋巴等），以最大化来源多样性
    - 优先**许可宽松 + 体量小**（几百 MB 级），下载成本低
    - 每个来源必须能定义为「恶性 vs 良性」，即存在可折叠的类别
    - 无法二分的数据集（如器官分类、分割掩膜）**排除**，不污染矩阵

用法：
    python scripts/61_build_source_matrix.py --plan          # 只列计划
    python scripts/61_build_source_matrix.py --download      # 下载并解压
    python scripts/61_build_source_matrix.py --report        # 汇报当前矩阵来源
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MS = ROOT / "data" / "raw" / "patch_multisource"
MANI = ROOT / "data" / "manifests"
UA = {"User-Agent": "Mozilla/5.0 (endo-wsi-research)"}

# ---------------------------------------------------------------- 定向候选
# 说明：malignant / benign 用于把类别名折叠成二分类；键名大小写与写法按 Kaggle 实际目录填写，
#       下载后由 --verify 校验是否匹配成功，不匹配会明确报警。
CANDIDATES: list[dict] = [
    {
        "ref": "andrewmvd/lung-and-colon-cancer-histopathological-images",
        "alias": "lung_colon",
        "organ": "lung+colon",
        "max_gb": 2.0,
        "malignant": {"lung_aca", "lung_scc", "colon_aca"},
        "benign": {"lung_n", "colon_n"},
        "note": "肺部与结肠：正常 vs 腺癌/鳞癌（5 类可折叠为 2 类）",
    },
    {
        "ref": "kmader/kidney-histopathology",
        "alias": "kidney",
        "organ": "kidney",
        "max_gb": 1.0,
        "malignant": {"kidney_tumor"},
        "benign": {"kidney_normal"},
        "note": "肾：正常 vs 肿瘤",
    },
    {
        "ref": "truthisneverlinear/brecahad",
        "alias": "brecahad",
        "organ": "breast",
        "max_gb": 1.5,
        "malignant": {"malignant", "1"},
        "benign": {"benign", "0"},
        "note": "乳腺：良恶性（与 BreakHis 不同来源，可互为对照）",
    },
]

# 已知可用的额外来源（若 Kaggle ref 失效，可用这些备选）
FALLBACKS: list[dict] = [
    {
        "ref": "ambarish/breakhis",
        "alias": "breakhis_all",
        "organ": "breast",
        "max_gb": 3.0,
        "malignant": {"malignant"},
        "benign": {"benign"},
        "note": "BreakHis 全放大倍数版（对照 400× 版）",
    },
]


def jget(url: str):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90) as r:
        return json.load(r)


def kaggle_view(ref: str) -> dict | None:
    try:
        return jget(f"https://www.kaggle.com/api/v1/datasets/view/{ref}")
    except Exception:  # noqa: BLE001
        return None


def find_class_dirs(root: Path) -> dict[str, int]:
    """统计目录树中「作为叶子目录且含图像」的目录名及图像数。"""
    counts: dict[str, int] = {}
    if not root.exists():
        return counts
    for d in root.rglob("*"):
        if not d.is_dir():
            continue
        n = len([p for p in d.glob("*")
                 if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff")])
        if n:
            counts[d.name] = counts.get(d.name, 0) + n
    return counts


def verify(cand: dict, root: Path) -> dict:
    counts = find_class_dirs(root)
    mal = sum(v for k, v in counts.items() if k in cand["malignant"])
    ben = sum(v for k, v in counts.items() if k in cand["benign"])
    return {"found_dirs": dict(sorted(counts.items(), key=lambda kv: -kv[1])[:12]),
            "n_malignant": mal, "n_benign": ben,
            "usable": bool(mal and ben)}


def download_and_extract(cand: dict) -> dict:
    alias = cand["alias"]
    dest = MS / alias
    zp = MS / f"{alias}.zip"
    if dest.exists() and any(dest.rglob("*.jpg")):
        return {"status": "already", **verify(cand, dest)}

    url = f"https://www.kaggle.com/api/v1/datasets/download/{cand['ref']}"
    cmd = [sys.executable, str(ROOT / "scripts" / "39_fetch_kaggle_zip.py"),
           "--ref", cand["ref"], "--segments", "8", "--workers", "8",
           "--out", str(MS.relative_to(ROOT))]
    print(f"  下载 {cand['ref']} …")
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), timeout=5400)
    ok = p.returncode == 0
    print(f"    {'OK' if ok else 'FAIL'} {(p.stdout or '')[-200:]}")

    # 39 号脚本按其自身规则命名 zip，这里统一为 alias
    got = None
    for z in MS.glob("*.zip"):
        if z.stem == cand["ref"].split("/")[-1]:
            got = z
            break
    if got is None and zp.exists():
        got = zp
    if got and got.stat().st_size > 0:
        dest.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(got) as z:
                z.extractall(dest)
            print(f"    解压 {len(list(z.namelist())) if False else ''} 完成 -> {dest.name}")
        except Exception as exc:  # noqa: BLE001
            print(f"    解压失败 {type(exc).__name__}: {exc}")
    return {"status": "downloaded" if ok else "failed", **verify(cand, dest)}


def report_matrix() -> dict:
    """汇报当前所有来源及其可用性（含已纳入矩阵的与未纳入的）。"""
    known = {
        "endo_histo": ("endometrium", {"EA"}, {"NE", "EH", "EP"}),
        "uterus_ca": ("uterus", {"tumour"}, {"non-tumour"}),
        "ovarian_ca": ("ovary",
                       {"Clear_Cell", "Endometri", "Mucinous", "Serous"}, {"Non_Cancerous"}),
        "breakhis_400x": ("breast", {"malignant"}, {"benign"}),
    }
    out: dict = {"in_matrix": {}, "candidates": {}}
    for name, (organ, mal, ben) in known.items():
        roots = [MS / name, MS / f"{name.replace('_400x', '-400x')}",
                 ROOT / "data" / "raw" / "endometrial_patch" / "extracted" / {
                     "endo_histo": "HIENet", "uterus_ca": "Uterus-Cancer"}.get(name, "")]
        got = None
        for r in roots:
            c = find_class_dirs(r)
            if sum(v for k, v in c.items() if k in mal) and sum(v for k, v in c.items() if k in ben):
                got = (r, c)
                break
        out["in_matrix"][name] = {"organ": organ, "usable": got is not None,
                                  "root": str(got[0].relative_to(ROOT)) if got else None,
                                  "dirs": dict(sorted(got[1].items(), key=lambda kv: -kv[1])[:8]) if got else {}}

    for cand in CANDIDATES + FALLBACKS:
        v = verify(cand, MS / cand["alias"])
        if v["usable"] or v["found_dirs"]:
            out["candidates"][cand["alias"]] = {"organ": cand["organ"], **v}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if not (args.plan or args.download or args.report):
        args.report = True

    MANI.mkdir(parents=True, exist_ok=True)

    if args.plan:
        print("=== 定向候选（要求能二分为恶性/良性）===")
        for c in CANDIDATES + FALLBACKS:
            info = kaggle_view(c["ref"])
            size = (info or {}).get("totalBytes") or (info or {}).get("totalBytesNullable")
            lic = (info or {}).get("licenseName") or (info or {}).get("licenseNameNullable")
            mb = f"{size/1024**2:.0f} MB" if size else "未知"
            print(f"  {c['alias']:14s} {c['organ']:12s} {mb:>10s}  {str(lic)[:18]:20s} "
                  f"恶={sorted(c['malignant'])} 良={sorted(c['benign'])}")
            print(f"                 {c['ref']}")

    if args.download:
        print("\n=== 下载与解压 ===")
        results = {}
        for c in CANDIDATES:
            info = kaggle_view(c["ref"])
            size = (info or {}).get("totalBytes") or (info or {}).get("totalBytesNullable") or 0
            if size and size > c["max_gb"] * 1024 ** 3:
                print(f"  [SKIP] {c['alias']} 超过体量上限（{size/1024**3:.2f} GB > {c['max_gb']} GB）")
                continue
            results[c["alias"]] = download_and_extract(c)
            print(f"    -> 可用={results[c['alias']].get('usable')} "
                  f"恶={results[c['alias']].get('n_malignant')} 良={results[c['alias']].get('n_benign')}")
        (MANI / "source_matrix_build_results.json").write_text(
            json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    rep = report_matrix()
    (MANI / "source_matrix_status.json").write_text(
        json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== 当前矩阵来源 ===")
    n_ok = 0
    for k, v in rep["in_matrix"].items():
        flag = "✅" if v["usable"] else "❌"
        n_ok += 1 if v["usable"] else 0
        print(f"  {flag} {k:16s} {v['organ']:12s} {v['dirs']}")
    print(f"  可用来源：{n_ok}")
    if rep["candidates"]:
        print("\n=== 已下载的候选 ===")
        for k, v in rep["candidates"].items():
            print(f"  {'✅' if v['usable'] else '⚠️ '} {k:14s} {v['organ']:12s} "
                  f"恶={v['n_malignant']} 良={v['n_benign']}  dirs={list(v['found_dirs'])[:6]}")
    print(f"\n-> {MANI/'source_matrix_status.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
