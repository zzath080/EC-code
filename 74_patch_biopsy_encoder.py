#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
74_patch_biopsy_encoder.py
==========================
给 `55_external_biopsy_validation.py` 加上 `--encoder`，使活检域外部验证也能用
第二编码器复现（对照实验的另一半）。

改动：
    1. FEAT_ROOT 支持从文件名推导编码器（新增 feat_root(encoder)）
    2. main() 里读 --encoder，并用它解析特征根与输出目录
    3. 输出目录按编码器区分，避免覆盖既有 phikon-v2 结果

用法：
    python scripts/74_patch_biopsy_encoder.py --dry-run
    python scripts/74_patch_biopsy_encoder.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "scripts" / "55_external_biopsy_validation.py"

REPLACEMENTS = [
    # 1) 特征根与输出目录改为按编码器解析
    ('FEAT_ROOT = ROOT / "data" / "features" / "phikon-v2"',
     'DEFAULT_ENCODER = "phikon-v2"\n'
     '\n'
     '\n'
     'def feat_root(encoder: str) -> Path:\n'
     '    return ROOT / "data" / "features" / encoder\n'
     '\n'
     '\n'
     'def out_dir_for(encoder: str) -> Path:\n'
     '    return ROOT / "runs" / ("_external_biopsy" if encoder == DEFAULT_ENCODER\n'
     '                            else f"_external_biopsy_{encoder}")\n'
     '\n'
     '\n'
     'FEAT_ROOT = feat_root(DEFAULT_ENCODER)   # 兼容旧调用'),

    # 2) main 里加参数
    ('    ap.add_argument("--max-instances", type=int, default=2048)',
     '    ap.add_argument("--max-instances", type=int, default=2048)\n'
     '    ap.add_argument("--encoder", default=DEFAULT_ENCODER)'),

    # 3) 用局部变量解析特征根与输出目录
    ('    # 源域：排除 01Z 诊断切片（保持"切除冰冻标本"的域定义一致）',
     '    feats = feat_root(args.encoder)\n'
     '    out = out_dir_for(args.encoder)\n'
     '    print(f"编码器 {args.encoder}；特征目录 {feats.relative_to(ROOT)}")\n'
     '    # 源域：排除 01Z 诊断切片（保持"切除冰冻标本"的域定义一致）'),

    ('        for sid, X in load_h5_dir(FEAT_ROOT / c).items():',
     '        for sid, X in load_h5_dir(feats / c).items():'),

    ('    tgt = load_h5_dir(FEAT_ROOT / TGT_COHORT)',
     '    tgt = load_h5_dir(feats / TGT_COHORT)'),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    text = TARGET.read_text(encoding="utf-8")
    applied = missed = 0
    for old, new in REPLACEMENTS:
        if old not in text:
            print(f"[MISS] {old[:70]}…")
            missed += 1
            continue
        if args.dry_run:
            print(f"[dry] {old[:66]}…")
        else:
            text = text.replace(old, new, 1)
            print(f"[OK ] {old[:66]}…")
        applied += 1

    # 4) 把 OUT 的写入点替换为局部 out
    if not args.dry_run:
        for a, b in [("    OUT.mkdir(parents=True, exist_ok=True)", "    out.mkdir(parents=True, exist_ok=True)"),
                     ("(OUT / \"results.json\")", "(out / \"results.json\")"),
                     ("with open(OUT / \"predictions.csv\"", "with open(out / \"predictions.csv\""),
                     ('(OUT / "report.md")', '(out / "report.md")'),
                     ("print(f\"\\n-> {OUT/'report.md'}\")", "print(f\"\\n-> {out/'report.md'}\")")]:
            if a in text:
                text = text.replace(a, b)
                print(f"[OK ] {a[:60]}…")
                applied += 1
        TARGET.write_text(text, encoding="utf-8")
        print(f"\n已写入 {TARGET.relative_to(ROOT)}")

    print(f"{'将' if args.dry_run else '已'}替换 {applied} 处，未匹配 {missed} 处")
    return 0 if missed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
