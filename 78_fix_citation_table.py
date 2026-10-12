"""修正 Introduction_Discussion 文档的参考文献表（一次性、可重复执行）。

为什么要写成脚本而不是手工编辑：
    该表格包含 20+ 行含 DOI、星号、方括号的长文本，手工 edit 的 old_string 极易
    因一个字符不匹配而失败（本会话已失败一次）。用行级定位 + 段落替换更稳。

做三件事：
    1. 用**本轮 Crossref/Europe PMC 核实过**的完整条目替换表中编号 1、8、9、18 四行；
    2. 把编号 8 从"gated 模型占位"（已删除）改为 Geirhos 捷径学习，
       因此正文基础模型引用 `[4–8]` 需改为 `[4–7]`；
    3. 在表末补一句"投稿前仍需处理"的说明。

用法：
    python scripts/78_fix_citation_table.py --dry-run
    python scripts/78_fix_citation_table.py
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "manuscript" / "Introduction_Discussion_可直接引用.md"

# 本轮核实过的完整条目（编号 → 整行内容）
VERIFIED = {
    1: ("| 1 | **Bray F, Laversanne M, Sung H, Ferlay J, et al. Global cancer statistics 2022: "
        "GLOBOCAN estimates of incidence and mortality worldwide for 36 cancers in 185 countries. "
        "*CA Cancer J Clin* 2024;74(3):229-263. doi:10.3322/caac.21834** | ✅ **本轮 Crossref 核实** |"),
    8: ("| 8 | **Geirhos R, Jacobsen J-H, Michaelis C, Zemel R, Brendel W, Bethge M, Wichmann FA. "
        "Shortcut learning in deep neural networks. *Nat Mach Intell* 2020;2(11):665-673. "
        "doi:10.1038/s42256-020-00257-z** | ✅ **本轮 Crossref 核实** |"),
    9: ("| 9 | **Zech JR, Badgeley MA, Liu M, Costa AB, Titano JJ, Oermann EK. Variable generalization "
        "performance of a deep learning model to detect pneumonia in chest radiographs: a cross-sectional "
        "study. *PLoS Med* 2018;15:e1002683. doi:10.1371/journal.pmed.1002683. PMID 30399157** | "
        "✅ **本轮 Europe PMC 核实** |"),
    18: ("| 18 | **Reinhard E, Adhikhmin M, Gooch B, Shirley P. Color transfer between images. "
         "*IEEE Comput Graph Appl* 2001;21(5):34-41. doi:10.1109/38.946629** | "
         "✅ **本轮 Crossref 核实**（原综述仅 snippet 级） |"),
}

# 正文里需要修正的引用范围：编号 8 已从"基础模型"改为 Geirhos，故 [4–8] → [4–7]
TEXT_FIXES = [
    ("[4–8]", "[4–7]"),
    ("[4-8]", "[4-7]"),
]

TAIL_NOTE = (
    "> ✅ **本轮新核实 4 条**：1（GLOBOCAN 2022）、8（Geirhos 捷径学习）、9（Zech 混杂变量）、\n"
    "> 18（Reinhard 颜色迁移）。其中 **8 和 9 是本文主线的核心引文，此前项目里没有**。\n"
    ">\n"
    "> ⚠️ **投稿前仍需处理**：① 本表编号与正文 `[n]` 需在合并全文时按 BSPC 要求"
    "（按出现顺序）重排一次；② 未核实的条目一律不得写入正式稿；\n"
    "> ③ 正文 §1.2 的基础模型引用已由 `[4–8]` 修正为 `[4–7]`（编号 8 现在是捷径学习，"
    "不再是基础模型）。\n"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not DOC.exists():
        print(f"[ERROR] 缺 {DOC}", file=sys.stderr)
        return 2
    lines = DOC.read_text(encoding="utf-8").splitlines()
    changes: list[str] = []

    # 1) 逐行替换已核实条目（按行首 "| N |" 精确定位）
    for num, new_line in VERIFIED.items():
        pat = re.compile(rf"^\|\s*{num}\s*\|")
        for i, ln in enumerate(lines):
            if pat.match(ln):
                if ln != new_line:
                    lines[i] = new_line
                    changes.append(f"表行 {num} 已替换")
                break
        else:
            changes.append(f"[WARN] 未找到表行 {num}")

    text = "\n".join(lines)

    # 2) 正文引用范围修正
    for old, new in TEXT_FIXES:
        if old in text:
            text = text.replace(old, new)
            changes.append(f"正文引用 {old} → {new}")

    # 3) 表末说明（幂等：已存在则不重复插入）
    if "本轮新核实 4 条" not in text:
        anchor = "> **投稿前必须处理**"
        if anchor in text:
            idx = text.index(anchor)
            end = text.index("\n\n", idx) if "\n\n" in text[idx:] else len(text)
            text = text[:end] + "\n\n" + TAIL_NOTE + text[end:]
            changes.append("已插入表末说明")
        else:
            text = text.rstrip() + "\n\n" + TAIL_NOTE
            changes.append("已追加表末说明（未找到原锚点）")

    if args.dry_run:
        print("\n".join(f"  [dry] {c}" for c in changes) or "  （无变化）")
        return 0

    DOC.write_text(text, encoding="utf-8")
    print("\n".join(f"  {c}" for c in changes) or "  （无变化）")
    print(f"\n-> {DOC.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
