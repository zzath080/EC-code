#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
24_test_encoder.py
==================
基础模型加载与前向烟雾测试 —— 在跑全量特征提取前先确认模型可用。

检查：
    - 权重文件大小是否与 HF 记录一致（防截断）
    - 能否用 transformers 加载
    - GPU 前向一次，输出维度与数值健康度
    - 显存占用与单批耗时（用于估算全量耗时）

用法：
    python scripts/24_test_encoder.py --encoder phikon-v2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / ".pylibs"
if LIB.exists():
    sys.path.insert(0, str(LIB))

ENCODERS = {
    "phikon-v2": {"path": ROOT / "models" / "phikon-v2", "repo": "owkin/phikon-v2", "expect_bytes": 1213455560},
    "vit-base-in": {"path": ROOT / "models" / "vit-base-in", "repo": "google/vit-base-patch16-224", "expect_bytes": None},
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="phikon-v2", choices=list(ENCODERS))
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    import torch
    from transformers import AutoModel

    spec = ENCODERS[args.encoder]
    wpath = spec["path"] / "model.safetensors"
    print(f"模型 {args.encoder}  目录 {spec['path']}")
    if not wpath.exists():
        print(f"[ERROR] 缺少权重 {wpath}", file=sys.stderr)
        return 2

    size = wpath.stat().st_size
    ok_size = True
    if spec["expect_bytes"]:
        ok_size = size == spec["expect_bytes"]
        print(f"权重大小 {size/1024**2:.1f} MB  期望 {spec['expect_bytes']/1024**2:.1f} MB  "
              f"{'✅ 一致' if ok_size else '❌ 不一致（可能被截断）'}")
    else:
        print(f"权重大小 {size/1024**2:.1f} MB")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    model = AutoModel.from_pretrained(str(spec["path"]))
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"加载完成 {time.time()-t0:.1f}s  设备 {device}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"参数量 {n_params/1e6:.1f} M")

    cfg = getattr(model, "config", None)
    if cfg is not None:
        print(f"hidden_size={getattr(cfg,'hidden_size',None)}  "
              f"image_size={getattr(cfg,'image_size',None)}  "
              f"patch_size={getattr(cfg,'patch_size',None)}")

    size_px = getattr(cfg, "image_size", 224) if cfg is not None else 224
    x = torch.randn(args.batch, 3, size_px, size_px, device=device)
    with torch.no_grad():
        t0 = time.time()
        out = model(x)
        if hasattr(out, "last_hidden_state"):
            emb = out.last_hidden_state[:, 0]
        elif hasattr(out, "pooler_output") and out.pooler_output is not None:
            emb = out.pooler_output
        else:
            emb = out[0][:, 0]
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0

    arr = emb.float().cpu().numpy()
    print(f"前向 batch={args.batch} -> {arr.shape}  用时 {dt*1000:.1f} ms  "
          f"({args.batch/dt:.1f} patch/s)")
    print(f"数值：mean={arr.mean():.4f}  std={arr.std():.4f}  "
          f"nan={int(np.isnan(arr).sum())}  全零行={int((np.abs(arr).sum(1)==0).sum())}")
    if device == "cuda":
        print(f"峰值显存 {torch.cuda.max_memory_allocated()/1024**3:.2f} GiB")

    est = 129000 / (args.batch / max(dt, 1e-9))
    print(f"\n估算：129,000 个 patch 约需 {est/60:.1f} 分钟（按当前吞吐）")

    if not ok_size:
        print("\n[STOP] 权重大小不符，特征结果不可信；请重新下载", file=sys.stderr)
        return 1
    print("\n✅ 编码器可用")
    return 0


if __name__ == "__main__":
    sys.exit(main())
