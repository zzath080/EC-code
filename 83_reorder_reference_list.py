"""把参考文献表按新的引用顺序重排（配合 scripts/82 的正文重排）。

输入：`Introduction_Discussion_可直接引用.md` 中现有文献表（含完整条目与核实方式）。
顺序：由正文实际出现顺序决定——从三个稿件文件中按
`Abstract -> Introduction -> Methods -> Results -> Discussion` 抽取 `[n]` 首次出现顺序。

设计要点：
    - **条目内容不重写**，只重排顺序并重编号，避免二次转录错误；
    - 重排后做双向完整性校验：正文引用的每个编号都要在表里存在，表里每条都要被正文引用；
    - 记录"旧号 -> 新号"映射，便于对照历史文档。

用法：
    python scripts/83_reorder_reference_list.py --dry-run
    python scripts/83_reorder_reference_list.py
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANU = ROOT / "docs" / "manuscript"
DOC = MANU / "Introduction_Discussion_可直接引用.md"
SCAN = ["Introduction_Discussion_可直接引用.md", "Methods_可直接引用.md",
        "Results_可直接引用.md"]

CITE_RE = re.compile(r"\[(\d+(?:\s*[,\u2013-]\s*\d+)*)\]")
ROW_RE = re.compile(r"^\|\s*(\d+)\s*\|\s*(.+?)\s*\|\s*(.*?)\s*\|\s*$")


def expand(g: str) -> list[int]:
    if "\u2013" in g or "-" in g:
        parts = [p.strip() for p in re.split(r"[,\u2013-]", g)]
        if len(parts) == 2 and all(p.isdigit() for p in parts):
            lo, hi = int(parts[0]), int(parts[1])
            return list(range(min(lo, hi), max(lo, hi) + 1))
    return [int(p) for p in g.split(",") if p.strip().isdigit()]


def citation_order() -> list[int]:
    seen: list[int] = []
    for f in SCAN:
        p = MANU / f
        if not p.exists():
            continue
        t = p.read_text(encoding="utf-8")
        # 剔除引用块（元说明里也会出现 [n]，不应计入正文顺序）
        t = re.sub(r"^>.*$", "", t, flags=re.M)
        for m in CITE_RE.finditer(t):
            for n in expand(m.group(1)):
                if n not in seen:
                    seen.append(n)
    return seen


def parse_table(text: str) -> dict[int, tuple[str, str]]:
    out: dict[int, tuple[str, str]] = {}
    for ln in text.splitlines():
        m = ROW_RE.match(ln)
        if m:
            out[int(m.group(1))] = (m.group(2), m.group(3))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    text = DOC.read_text(encoding="utf-8")
    table = parse_table(text)
    order = citation_order()

    if not table:
        print("[ERROR] 未解析到文献表", file=sys.stderr)
        return 2

    cited = set(order)
    listed = set(table)
    only_listed = sorted(listed - cited)
    only_cited = sorted(cited - listed)
    if only_cited:
        print(f"[ERROR] 正文引用了表中不存在的编号：{only_cited}", file=sys.stderr)
        return 3
    if only_listed:
        print(f"[WARN] 表中存在未被正文引用的条目：{only_listed}（将被移除）")

    # 新编号 = 出现顺序
    old_to_new = {old: i + 1 for i, old in enumerate(order)}
    lines = ["| # | 文献 | 核实方式 |", "| --- | --- | --- |"]
    for old in order:
        body, how = table[old]
        lines.append(f"| {old_to_new[old]} | {body} | {how} |")
    new_table = "\n".join(lines)

    # 替换旧表
    start = text.index("| # | 文献 | 核实方式 |")
    end = text.index("\n\n", text.index("|", text.index("\n", start) + 1))
    # 找到表区间末尾：连续以 | 开头的行
    tbl_lines = []
    for ln in text[start:].splitlines():
        if ln.startswith("|"):
            tbl_lines.append(ln)
        elif tbl_lines:
            break
    old_block = "\n".join(tbl_lines)
    new_text = text.replace(old_block, new_table, 1)

    if args.dry_run:
        print(f"[dry] 将重排 {len(order)} 条；映射（旧->新）：{old_to_new}")
        print(new_table[:500])
        return 0

    DOC.write_text(new_text, encoding="utf-8")
    back = DOC.read_text(encoding="utf-8")
    assert back == new_text, "写回不一致"
    print(f"已重排 {len(order)} 条文献；UTF-8 往返校验通过")
    print(f"映射（旧->新）：{old_to_new}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
