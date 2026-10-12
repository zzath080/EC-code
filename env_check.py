#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
env_check.py — 环境自检
=======================
检查工作区内 `.pylibs` 的关键依赖是否可用，并报告 GPU 状态。
用法：python scripts/env_check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / ".pylibs"
if LIB.exists():
    sys.path.insert(0, str(LIB))

# 必需依赖（与 requirements.txt 对应：pip install -r requirements.txt）
MODULES = ["numpy", "pandas", "sklearn", "scipy", "PIL", "tifffile",
           "openslide", "openslide_bin", "h5py", "safetensors", "tokenizers",
           "transformers", "torch", "torchvision", "cv2", "matplotlib"]
# 可选依赖：全仓库**没有任何脚本直接 import**（已用 grep 核实），
# 仅作信息展示。MISS 不影响主链路，不必为了消掉这两行而安装。
OPTIONAL = ["zarr", "timm"]

print(f"python {sys.version.split()[0]}")
print(f"pylibs {LIB}  exists={LIB.exists()}")
if not LIB.exists():
    print("  （.pylibs 是原开发环境的依赖目录；此处不存在属正常，请用 requirements.txt 自建环境）")
print()
for name in MODULES:
    try:
        mod = __import__(name)
        version = getattr(mod, "__version__", "?")
        print(f"  {name:16s} OK    {version}")
    except Exception as exc:  # noqa: BLE001
        print(f"  {name:16s} MISS  {type(exc).__name__}   <- 必需，请安装")
for name in OPTIONAL:
    try:
        mod = __import__(name)
        version = getattr(mod, "__version__", "?")
        print(f"  {name:16s} OK    {version}  (可选)")
    except Exception:  # noqa: BLE001
        print(f"  {name:16s} MISS  (可选，可忽略)")

print()
try:
    import torch
    print(f"torch {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device: {torch.cuda.get_device_name(0)}")
        print(f"memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GiB")
        x = torch.randn(1024, 1024, device="cuda")
        y = (x @ x).sum().item()
        print(f"matmul smoke test OK ({y:.1f})")
except Exception as exc:  # noqa: BLE001
    print(f"torch 检查失败: {type(exc).__name__}: {exc}")
