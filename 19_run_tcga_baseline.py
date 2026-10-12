#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
19_run_tcga_baseline.py
=======================
一键跑通 TCGA 侧的端到端基线（Stage 1→2→3），并在依赖缺失时给出明确提示。

流程：
    1. 检查依赖（torch / transformers / phikon-v2 权重 / patch 索引）
    2. 对缺失特征的队列执行 patch 索引（若需要）
    3. 用 Phikon-v2 提取切片级特征（HDF5）
    4. 生成患者级划分清单
    5. 训练 ABMIL / mean / max 三种聚合器并输出指标（AUC + CI + ECE + Brier）

用法：
    # 先看依赖是否齐备
    python scripts/19_run_tcga_baseline.py --check

    # 全流程
    python scripts/19_run_tcga_baseline.py --cohort tcga_pair

    # 只跑某一步
    python scripts/19_run_tcga_baseline.py --only features
    python scripts/19_run_tcga_baseline.py --only train
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / ".pylibs"
if LIB.exists():
    sys.path.insert(0, str(LIB))

PY = sys.executable


def run(cmd: list[str]) -> int:
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT))
    return proc.returncode


def check_deps() -> dict:
    deps: dict[str, object] = {}

    def try_import(name: str):
        try:
            mod = __import__(name)
            return getattr(mod, "__version__", "?")
        except Exception as exc:  # noqa: BLE001
            return f"MISSING ({type(exc).__name__})"

    for m in ("torch", "transformers", "safetensors", "h5py", "openslide"):
        deps[m] = try_import(m)

    phikon = ROOT / "models" / "phikon-v2" / "model.safetensors"
    deps["phikon_v2_weights"] = (f"{phikon.stat().st_size/1024**3:.2f} GB"
                                 if phikon.exists() else "MISSING")

    cuda = "n/a"
    try:
        import torch
        cuda = f"{torch.cuda.is_available()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})"
    except Exception:  # noqa: BLE001
        pass
    deps["cuda"] = cuda

    for cohort in ("tcga_normal", "tcga_tumor"):
        d = ROOT / "data" / "patches" / cohort
        deps[f"patches_{cohort}"] = len(list(d.glob("*.csv"))) if d.exists() else 0

    for cohort in ("tcga_normal", "tcga_tumor"):
        d = ROOT / "data" / "features" / "phikon-v2" / cohort
        deps[f"features_{cohort}"] = len(list(d.glob("*.h5"))) if d.exists() else 0

    return deps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="tcga_pair")
    ap.add_argument("--encoder", default="phikon-v2")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--only", choices=["patches", "features", "split", "train"], default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--wsi-dirs", nargs="*",
                    default=["data/raw/tcga_ucec_normals", "data/raw/tcga_ucec_tumor_subset", "TCGA-UCEC"])
    args = ap.parse_args()

    print("=" * 72)
    print("TCGA 端到端基线 runner")
    print("=" * 72)
    deps = check_deps()
    print(json.dumps(deps, indent=2, ensure_ascii=False))

    blockers = []
    if str(deps.get("torch", "")).startswith("MISSING"):
        blockers.append("torch 未安装（用 scripts/pipx.py 安装，见 README §5.1）")
    if str(deps.get("transformers", "")).startswith("MISSING"):
        blockers.append("transformers 未安装")
    if deps.get("phikon_v2_weights") == "MISSING":
        blockers.append("Phikon-v2 权重缺失（python scripts/11_fetch_models.py --model phikon-v2）")
    if not deps.get("patches_tcga_normal"):
        blockers.append("tcga_normal 的 patch 索引缺失（先跑 scripts/10_tiling.py）")

    if args.check:
        print("\n阻塞项：" + ("无 ✅" if not blockers else ""))
        for b in blockers:
            print(f"  - {b}")
        return 0 if not blockers else 1

    if blockers:
        print("\n[STOP] 依赖未齐备，无法执行：")
        for b in blockers:
            print(f"  - {b}")
        return 2

    steps: list[tuple[str, list[str]]] = [
        ("patches", [PY, "scripts/10_tiling.py", "--input", "data/raw/tcga_ucec_normals",
                     "--cohort", "tcga_normal", "--no-verify"]),
        ("patches", [PY, "scripts/10_tiling.py", "--input", "data/raw/tcga_ucec_tumor_subset",
                     "--cohort", "tcga_tumor", "--no-verify"]),
        ("features", [PY, "scripts/12_extract_features.py", "--patches", "data/patches/tcga_normal",
                      "--cohort", "tcga_normal", "--encoder", args.encoder, "--save",
                      "--wsi-dirs", *args.wsi_dirs]),
        ("features", [PY, "scripts/12_extract_features.py", "--patches", "data/patches/tcga_tumor",
                      "--cohort", "tcga_tumor", "--encoder", args.encoder, "--save",
                      "--wsi-dirs", *args.wsi_dirs]),
        ("split", [PY, "scripts/14_build_split_manifest.py", "--cohort", args.cohort]),
    ]

    for name, cmd in steps:
        if args.only and args.only != name:
            continue
        rc = run(cmd)
        if rc != 0:
            print(f"[WARN] 步骤 {name} 返回 {rc}")

    if not args.only or args.only == "train":
        # 把两个队列的特征合并成训练可用的清单
        merge_cmd = [PY, "scripts/20_merge_feature_manifest.py", "--cohort", args.cohort]
        run(merge_cmd)
        for agg in ("mean", "max", "abmil", "clam"):
            run([PY, "scripts/13_mil_train_eval.py",
                 "--features", f"data/features/{args.encoder}",
                 "--manifest", f"data/manifests/{args.cohort}_split_manifest.csv",
                 "--aggregator", agg, "--epochs", str(args.epochs)])

    print("\n完成。指标见 runs/<run_id>/metrics.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
