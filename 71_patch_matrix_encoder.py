#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
71_patch_matrix_encoder.py
==========================
把 60_cross_source_matrix.py 的 cmd_eval 主体改成使用**按编码器解析出的输出目录**，
并加上 --encoder 参数（支持第二编码器对照）。

做法：把 cmd_eval 内部残留的模块级 OUT 用法替换为局部 out，
并在 argparse 里注入 --encoder，同时把 encoder 传给 cmd_extract / cmd_eval。

用法：
    python scripts/71_patch_matrix_encoder.py --dry-run
    python scripts/71_patch_matrix_encoder.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "scripts" / "60_cross_source_matrix.py"

REPLACEMENTS = [
    # cmd_eval 主体里的输出目录：改用局部 out
    ("    OUT.mkdir(parents=True, exist_ok=True)\n    (OUT / \"results.json\")",
     "    out.mkdir(parents=True, exist_ok=True)\n    (out / \"results.json\")"),
    ("with open(OUT / \"matrix_auc.csv\"", "with open(out / \"matrix_auc.csv\""),
    ("(OUT / \"report.md\").write_text", "(out / \"report.md\").write_text"),
    ("print(f\"\\n-> {OUT/'report.md'}\")", "print(f\"\\n-> {out/'report.md'}\")"),
    ("print(f\"-> {OUT/'matrix_auc.csv'}\")", "print(f\"-> {out/'matrix_auc.csv'}\")"),
    # 传参
    ("rc = cmd_extract(args.batch, args.limit)",
     "rc = cmd_extract(args.batch, args.limit, args.encoder)"),
    ("return cmd_eval(args.seed)", "return cmd_eval(args.seed, args.encoder)"),
    # argparse 注入 --encoder
    ('    ap.add_argument("--seed", type=int, default=42)',
     '    ap.add_argument("--seed", type=int, default=42)\n'
     '    ap.add_argument("--encoder", default=DEFAULT_ENCODER, choices=list(ENCODER_SPECS))'),
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

    if not args.dry_run and applied:
        TARGET.write_text(text, encoding="utf-8")
        print(f"\n已写入 {TARGET.relative_to(ROOT)}")
    print(f"{'将' if args.dry_run else '已'}替换 {applied} 处，未匹配 {missed} 处")
    return 0 if missed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
