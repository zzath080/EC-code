"""按最终引用顺序重建文献表（修正"正文已重排、表未重排"的错位）。

事故背景：
    `scripts/82` 只在**引用块之外**改写正文引用号，而文献表用 `| n |` 表格形式表达编号，
    因此重排后出现**正文与表格大面积错位**：正文 `[8]` 语境是 van den Berg，
    表中 `[8]` 却是 Geirhos。`scripts/83` 因解析问题成为空操作。
    本脚本改为**按内容索引条目**（而非按编号），再按最终顺序重建表格，从根上避免错位。

做法：
    1. 从现有表格解析出 (编号 -> 条目, 核实方式)，并为每条生成稳定的内容键
       （第一作者姓氏 + 关键判别词，用于区分同姓条目，如两篇 Lu）；
    2. 用 EXpected_ORDER（人工核定的最终顺序）取出条目；
    3. 重建表格，编号 = 序号；同时做双向完整性校验。

用法：
    python scripts/85_rebuild_reference_table.py --dry-run
    python scripts/85_rebuild_reference_table.py
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANU = ROOT / "docs" / "manuscript"
DOC = MANU / "Introduction_Discussion_可直接引用.md"

ROW_RE = re.compile(r"^\|\s*(\d+)\s*\|\s*(.+?)\s*\|\s*(.*?)\s*\|\s*$")

# 最终顺序 = 正文首次出现顺序（Introduction 的 [1..19]，再是 Methods 的 [20],[21]）。
# 每项用「第一作者姓氏 + 判别词」定位，判别词可为 None（姓氏唯一时）。
FINAL_ORDER: list[tuple[str, str | None]] = [
    ("Bray", "GLOBOCAN"),
    ("Mohammadi", None),
    ("Thiringer", None),
    ("Chen", "general-purpose"),      # UNI
    ("Lu", "visual-language"),        # CONCH
    ("Xu", None),                     # Virchow
    ("Filiot", None),                 # Phikon-v2
    ("van den Berg", None),           # EC 基础模型平台期  -> [8]
    ("Geirhos", None),                # 捷径学习          -> [9]
    ("Zech", None),                   # 混杂变量          -> [10]
    ("Tellez", None),                 #                   -> [11]
    ("Jahanifar", None),              #                   -> [12]
    ("Boschman", None),               #                   -> [13]
    ("Campanella", None),             #                   -> [14]
    ("Neidlinger", None),             #                   -> [15]
    ("Zamanitajeddin", None),         #                   -> [16]
    ("Aubreville", None),             # MIDOG             -> [17]
    ("Gelton", None),                 # ENDO-AID 数据     -> [18]
    ("Wagner", None),                 # 结直肠跨标本有解  -> [19]
    ("Ilse", None),                   # ABMIL             -> [20]
    ("Lu", "Data-efficient"),         # CLAM              -> [21]
]


# 若某条目在现有表中已被误删（本会话发生过：`scripts/83` 因编号错位把 Wagner 判为
# "表中未引用"而删除），在此补录完整条目，避免重建失败。
MISSING_ENTRIES: dict[str, tuple[str, str]] = {
    "Wagner": (
        "Wagner SJ, Reisenbüchler D, West NP, et al. Transformer-based biomarker prediction "
        "from colorectal cancer histology. *Cancer Cell* 2023;41(9):1650-1661. "
        "doi:10.1016/j.ccell.2023.08.002",
        "既有综述 `[VERIFIED]`",
    ),
}


def parse_table(text: str) -> list[tuple[int, str, str]]:
    out = []
    for ln in text.splitlines():
        m = ROW_RE.match(ln)
        if m:
            out.append((int(m.group(1)), m.group(2).strip(), m.group(3).strip()))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    text = DOC.read_text(encoding="utf-8")
    rows = parse_table(text)
    if not rows:
        print("[ERROR] 未解析到文献表", file=sys.stderr)
        return 2

    # 内容索引
    pool = list(rows)
    chosen: list[tuple[str, str]] = []
    for surname, discriminator in FINAL_ORDER:
        hit = None
        for i, (_n, body, how) in enumerate(pool):
            if surname.lower() not in body.lower():
                continue
            if discriminator and discriminator.lower() not in body.lower():
                continue
            hit = (i, body, how)
            break
        if hit is None:
            if surname in MISSING_ENTRIES:
                chosen.append(MISSING_ENTRIES[surname])
                print(f"[INFO] {surname} 不在表中，已用 MISSING_ENTRIES 补录")
                continue
            print(f"[ERROR] 在表中找不到条目：{surname}"
                  f"{' / ' + discriminator if discriminator else ''}", file=sys.stderr)
            return 3
        pool.pop(hit[0])
        chosen.append((hit[1], hit[2]))

    if pool:
        left = [r[1][:60] for r in pool]
        print(f"[WARN] 有 {len(pool)} 条未被最终顺序收录，将移除：")
        for x in left:
            print("   -", x)

    new_table_lines = ["| # | 文献 | 核实方式 |", "| --- | --- | --- |"]
    for i, (body, how) in enumerate(chosen, 1):
        new_table_lines.append(f"| {i} | {body} | {how} |")
    new_table = "\n".join(new_table_lines)

    # 定位并替换旧表（连续的以 | 开头的行，且首行为表头）
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("| # | 文献"))
    end = start
    while end < len(lines) and lines[end].startswith("|"):
        end += 1
    old_block = "\n".join(lines[start:end])
    new_text = text.replace(old_block, new_table, 1)

    if args.dry_run:
        print(f"[dry] 将重建 {len(chosen)} 条；旧表 {end - start} 行 -> 新表 {len(new_table_lines)} 行")
        print(new_table[:700])
        return 0

    DOC.write_text(new_text, encoding="utf-8")
    back = DOC.read_text(encoding="utf-8")
    assert back == new_text, "写回不一致"
    print(f"已重建文献表：{len(chosen)} 条；UTF-8 往返校验通过")
    for i, (body, _how) in enumerate(chosen, 1):
        print(f"  [{i:>2}] {body[:72]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
