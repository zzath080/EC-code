#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
72_bin_to_safetensors.py
========================
把 `pytorch_model.bin` 转成 `model.safetensors`。

原因：transformers 5.17 出于 CVE-2025-32434 拒绝用 `torch<2.6` 加载 `.bin`
（报 "we now require users to upgrade torch to at least v2.6"）。
本机 torch 是 2.5.1+cu124（2.4 GB 的包，不值得为这一个模型重下），
而 safetensors 不受该限制。因此离线转换一次即可。

用法：
    python scripts/72_bin_to_safetensors.py --model models/phikon
    python scripts/72_bin_to_safetensors.py --model models/phikon --verify
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / ".pylibs"
if LIB.exists():
    sys.path.insert(0, str(LIB))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="模型目录（相对或绝对）")
    ap.add_argument("--verify", action="store_true", help="转换后立即尝试加载验证")
    args = ap.parse_args()

    import torch
    from safetensors.torch import save_file

    d = Path(args.model)
    if not d.is_absolute():
        d = ROOT / d
    src = d / "pytorch_model.bin"
    dst = d / "model.safetensors"
    if not src.exists():
        print(f"[ERROR] 找不到 {src}", file=sys.stderr)
        return 2
    if dst.exists():
        print(f"已存在 {dst}（{dst.stat().st_size/1024**2:.1f} MB），跳过转换")
        return 0

    print(f"加载 {src.name}（{src.stat().st_size/1024**2:.1f} MB）…")
    obj = torch.load(str(src), map_location="cpu", weights_only=False)
    # 兼容三种常见结构
    if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
        sd = obj["state_dict"]
    elif isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict):
        sd = obj["model"]
    elif isinstance(obj, dict):
        sd = obj
    else:
        print(f"[ERROR] 无法识别的结构：{type(obj)}", file=sys.stderr)
        return 2

    # safetensors 只接受 tensor；非 tensor 条目单独报告并丢弃
    tensors, dropped = {}, []
    for k, v in sd.items():
        if isinstance(v, torch.Tensor):
            tensors[k] = v.detach().clone()
        else:
            dropped.append(k)
    print(f"张量 {len(tensors)} 个，非张量条目 {len(dropped)} 个（已丢弃）")
    if dropped:
        print("  丢弃示例：", dropped[:5])

    # 去掉 transformers 常见前缀
    pref = {k.split(".")[0] for k in tensors}
    if len(pref) == 1 and next(iter(pref)) in ("module", "model", "net"):
        p = next(iter(pref))
        tensors = {k[len(p) + 1:]: v for k, v in tensors.items()}
        print(f"  去掉前缀 '{p}.'")

    save_file(tensors, str(dst))
    print(f"已写出 {dst.name}（{dst.stat().st_size/1024**2:.1f} MB）")
    print(f"原 .bin 保留在 {src.name}（不删除，便于核对）")

    if args.verify:
        from transformers import AutoModel
        m = AutoModel.from_pretrained(str(d)).eval()
        print(f"验证：加载成功，hidden={m.config.hidden_size}，"
              f"层数={m.config.num_hidden_layers}，"
              f"参数={sum(p.numel() for p in m.parameters())/1e6:.1f}M")
    return 0


if __name__ == "__main__":
    sys.exit(main())
