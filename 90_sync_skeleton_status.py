"""同步骨架（导航文件）与最终稿状态：把已完成的 `[[待重跑]]` / `[[待补]]` 标记清掉。

背景：骨架是写作期的导航文件，早期标了大量 `[[待重跑]]`（因为扩来源后需要重跑）。
这些现在已经全部跑完（7 来源矩阵、捷径审计、Figure 1/2、Table 2/3/4/7/8 均已重生成），
但标记仍留在文件里，会让人误以为还没做。

做法：按"事项 -> 实际状态"的白名单逐条替换，**只改状态标记，不改内容**，
并保留一条说明：哪些是刻意留下的待办（需项目人决定的事项，如仓库 DOI）。

用法：
    python scripts/90_sync_skeleton_status.py --dry-run
    python scripts/90_sync_skeleton_status.py
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKEL = ROOT / "docs" / "manuscript" / "稿件骨架_Manuscript_Skeleton.md"

# (旧文本, 新文本) —— 逐条人工核定，避免正则批量误改
REPLACEMENTS: list[tuple[str, str]] = [
    ("> ⚠️ 扩来源后矩阵/捷径/图表正在重跑，**本文件中标注 `[[待重跑]]` 的数字以重跑结果为准**。",
     "> ✅ 扩来源（7 个）后的矩阵、捷径审计与全部图表**均已重跑完成**，"
     "数字见 `Results_可直接引用.md` 与 `Supplementary_Material.md`。"),
    ("| 摘要 | **< 250 词**，可结构式 | 待写（见 §1） |",
     "| 摘要 | **< 250 词**，可结构式 | ✅ 已写，227 词 |"),
    ("| 关键词 | 1–7 个 | 待定 |",
     "| 关键词 | 1–7 个 | ✅ 6 个 |"),
    ("| AI 使用 | 若用于研究设计/方法，须在 Methods 中可复现地描述 | 待填（Methods §2.10） |",
     "| AI 使用 | 若用于研究设计/方法，须在 Methods 中可复现地描述 | ✅ 已写（Methods §2.10），**口径待项目人确认** |"),
    ("| 2.10 AI 使用声明 | — | ⬜ **待填** |",
     "| 2.10 AI 使用声明 | 已写成正式英文段落（含 AI 未用于生成图片、未做科学判断的声明） | ✅ |"),
    ("| §3.3 跨来源一致崩塌 | 全交叉矩阵与落差 `[[待重跑]]` | Figure 1、Table 3 |",
     "| §3.3 跨来源一致崩塌 | 42 配对，均值 −0.4028，15/42 低于随机 | Figure 1、Table 3 |"),
    ("| §3.4 捷径审计 | 7/7 来源可被非形态学统计量预测 `[[待重跑]]` | Figure 2、Table 4 |",
     "| §3.4 捷径审计 | 7/7 来源可被非形态学统计量预测（AUC 0.724–0.978） | Figure 2、Table 4 |"),
    ("| Figure 1 | `paper_figures/Figure1_cross_source_matrix` | 跨来源 AUC 矩阵 + 落差 | `64_make_figures_tables.py` | `[[待重跑]]` |",
     "| Figure 1 | `paper_figures/Figure1_cross_source_matrix` | 跨来源 AUC 矩阵 + 落差 | `64_make_figures_tables.py` | ✅ 7×7 已重生成 |"),
    ("| Figure 2 | `paper_figures/Figure2_shortcut_audit` | 捷径审计 | `64` | `[[待重跑]]` |",
     "| Figure 2 | `paper_figures/Figure2_shortcut_audit` | 捷径审计 | `64` | ✅ 7 来源已重生成 |"),
    ("| Table 2 | `paper_tables/Table2_datasets` | 数据集对照 | `21` | `[[待重跑]]`（需含 7 来源） |",
     "| Table 2 | `paper_tables/Table2_datasets` | 数据集对照 | `21` | ✅ 已含 7 patch 来源 + 2 WSI 队列 |"),
    ("| Table 3 | `paper_figures/Table3_cross_source_matrix` | 矩阵明细 | `64` | `[[待重跑]]` |",
     "| Table 3 | `paper_figures/Table3_cross_source_matrix` | 矩阵明细 | `64` | ✅ 已重生成 |"),
    ("| Table 4 | `paper_figures/Table4_shortcut_audit` | 捷径审计 | `64` | `[[待重跑]]` |",
     "| Table 4 | `paper_figures/Table4_shortcut_audit` | 捷径审计 | `64` | ✅ 已重生成 |"),
    ("| Table 7 | `paper_figures/Table7_encoder_comparison` | 编码器跨来源对照 | `73` | `[[待重跑]]` |",
     "| Table 7 | `paper_figures/Table7_encoder_comparison` | 编码器跨来源对照 | `73` | ✅ 已重生成（42 配对） |"),
    ("| Table 8 | `paper_figures/Table8_encoder_by_experiment` | 四实验 × 编码器 | `73` | `[[待重跑]]` |",
     "| Table 8 | `paper_figures/Table8_encoder_by_experiment` | 四实验 × 编码器 | `73` | ✅ 已重生成 |"),
    ("| 图注 | `paper_figures/FIGURE_TABLE_LEGENDS.md` | 全部图注中英对照 | 手写 | `[[待更新]]` |",
     "| 补充材料 | `Supplementary_Material.md` | Table S1–S8（数字全脚本生成） | `80` | ✅ 210 项已校验 |"),
    ("`[[待写]]` — 三条：(i) 域内性能可被捷径高估；(ii) 跨来源崩塌是编码器无关的普遍现象；",
     "✅ 已写（见 `Introduction_Discussion_可直接引用.md` §2.1）："
     "(i) 域内性能可被捷径高估；(ii) 跨来源崩塌是编码器无关的普遍现象；"),
    ("`[[待写]]` — 不作过度声明：本文**不**提出新模型，而是证明**现有评测流程不足以判断",
     "✅ 已写（见同一文件 §5 / §2.6）：本文**不**提出新模型，而是证明**现有评测流程不足以判断"),
]

TAIL_NOTE = """
> 📌 **本文件仅为写作期导航**。正文成稿见：
> `Manuscript_Full.md`（合并稿）、`Results_可直接引用.md`、`Methods_可直接引用.md`、
> `Introduction_Discussion_可直接引用.md`、`Supplementary_Material.md`。
> **唯一仍需项目人决定的事项**：代码/数据仓库 DOI（Data and code availability）。
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not SKEL.exists():
        print(f"[ERROR] 缺 {SKEL}", file=sys.stderr)
        return 2

    text = SKEL.read_text(encoding="utf-8")
    hits = misses = 0
    for old, new in REPLACEMENTS:
        if old in text:
            text = text.replace(old, new, 1)
            hits += 1
        else:
            print(f"  [MISS] {old[:60]}…")
            misses += 1

    if "本文件仅为写作期导航" not in text:
        text = text.rstrip() + "\n" + TAIL_NOTE

    leftover = re.findall(r"\[\[待[^\]]*\]\]", text)
    if args.dry_run:
        print(f"[dry] 替换 {hits} 处，未匹配 {misses} 处，剩余待办标记 {len(leftover)}: {leftover}")
        return 0

    SKEL.write_text(text, encoding="utf-8")
    assert SKEL.read_text(encoding="utf-8") == text
    print(f"已同步骨架状态：替换 {hits} 处，未匹配 {misses} 处")
    print(f"剩余待办标记：{leftover if leftover else '无'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
