#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""check_fonts.py — 检查可视化环境与可用字体（论文图建议用英文字体，避免中文乱码）。"""
from __future__ import annotations
import sys
import matplotlib
import matplotlib.font_manager as fm

print("python", sys.version.split()[0])
print("matplotlib", matplotlib.__version__)
names = {f.name for f in fm.fontManager.ttflist}
for c in ["SimHei", "Microsoft YaHei", "SimSun", "DejaVu Sans", "Arial", "Helvetica"]:
    print(f"  font {c:20s} {'OK' if c in names else 'MISS'}")
# 论文图将以英文标注为主，DejaVu Sans 是 matplotlib 自带，必定可用
print("default family:", matplotlib.rcParams["font.family"])
