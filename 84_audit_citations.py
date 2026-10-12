"""审计「正文 [n] 的上下文」与「文献表第 n 条」是否指向同一工作。

为什么需要这个审计（真实事故）：
    参考文献重排时，正文引用号被改写、但文献表没有被同步改写，导致两者**大面积错位**——
    例如正文 `[8]` 语境是 van den Berg 的 EC 基础模型平台期，而表中 `[8]` 却是 Geirhos 的
    捷径学习。这类错位在投稿中是致命的，且**肉眼看不出**（编号本身完全合法）。

做法：
    1. 从文献表取第 n 条的第一作者姓氏（或数据集/机构名）；
    2. 找到正文中每个 `[n]` 的位置，检查其**同一段落/邻近文本**是否出现该条目对应的
       可识别关键词（作者姓、或该工作的特征词，如 "GLOBOCAN"、"shortcut learning"）；
    3. 报告无法在邻近文本中建立联系的 (编号, 条目) 对，供人工判断。

注意：这是**启发式**审计，会存在假阳性（例如引文紧邻但没有重复关键词）。
因此报告为"需人工确认"而不是"错误"。真正的完整性检查是 scripts/83 的双向校验。

用法：
    python scripts/84_audit_citations.py
退出码 0 = 无可疑项；1 = 存在需人工确认的项。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANU = ROOT / "docs" / "manuscript"
DOC = MANU / "Introduction_Discussion_可直接引用.md"
SCAN = ["Introduction_Discussion_可直接引用.md", "Methods_可直接引用.md"]

ROW_RE = re.compile(r"^\|\s*(\d+)\s*\|\s*(.+?)\s*\|\s*(.*?)\s*\|\s*$")

# 手工登记的"关键词 -> 期望条目"锚点：正文里这些话题必须引到对应文献。
# 只登记**高置信**的锚点，避免假阳性淹没真问题。
ANCHORS = [
    ("GLOBOCAN",                 "Bray"),
    ("shortcut",                 "Geirhos"),
    ("metal tokens",             "Zech"),
    ("plateau",                  "van den Berg"),
    ("endoscopic biopsies",      "Wagner"),
    ("MIDOG",                    "Aubreville"),
    ("attention-based MIL",      "Ilse"),
    ("clustering-constrained",   "Lu"),
    ("Reinhard",                 "Reinhard"),
    ("ENDO-AID",                 "Gelton"),
    ("scanning",                 "Thiringer"),
]


def parse_table(text: str) -> dict[int, str]:
    out = {}
    for ln in text.splitlines():
        m = ROW_RE.match(ln)
        if m:
            out[int(m.group(1))] = m.group(2)
    return out


def first_author_surname(entry: str) -> str | None:
    e = entry.strip().lstrip("*").strip()
    m = re.match(r"([A-Z][A-Za-z\u00c0-\u024f'\-]+)", e)
    return m.group(1) if m else None


def main() -> int:
    text = DOC.read_text(encoding="utf-8")
    table = parse_table(text)
    if not table:
        print("[ERROR] 未解析到文献表", file=sys.stderr)
        return 2

    print(f"文献表 {len(table)} 条")
    problems: list[str] = []

    # ---- 1) 双向完整性：正文引用的编号是否都在表里
    cited: set[int] = set()
    for f in SCAN:
        p = MANU / f
        if not p.exists():
            continue
        t = re.sub(r"^>.*$", "", p.read_text(encoding="utf-8"), flags=re.M)
        for m in re.finditer(r"\[(\d+(?:\s*[,\u2013-]\s*\d+)*)\]", t):
            g = m.group(1)
            if "\u2013" in g or "-" in g:
                parts = [x.strip() for x in re.split(r"[,\u2013-]", g)]
                if len(parts) == 2 and all(x.isdigit() for x in parts):
                    cited.update(range(int(parts[0]), int(parts[1]) + 1))
                    continue
            cited.update(int(x) for x in g.split(",") if x.strip().isdigit())

    only_listed = sorted(set(table) - cited)
    only_cited = sorted(cited - set(table))
    if only_cited:
        problems.append(f"正文引用了表中不存在的编号：{only_cited}")
    if only_listed:
        problems.append(f"表中存在未被正文引用的条目：{only_listed}")

    # ---- 2) 锚点语义检查：话题附近是否引到了期望文献
    new_of = {}
    for n, entry in table.items():
        sn = first_author_surname(entry)
        if sn:
            new_of[sn] = n
    for keyword, expected_surname in ANCHORS:
        num = new_of.get(expected_surname)
        if num is None:
            problems.append(f"锚点「{keyword}」期望的 {expected_surname} 不在表里")
            continue
        # 在正文中找该关键词，取其前后 400 字窗口内的引用号
        for m in re.finditer(re.escape(keyword), text, flags=re.I):
            win = text[max(0, m.start() - 400): m.end() + 400]
            nums: set[int] = set()
            for mm in re.finditer(r"\[(\d+(?:\s*[,\u2013-]\s*\d+)*)\]", win):
                g = mm.group(1)
                if "\u2013" in g or "-" in g:
                    if int(g.split("\u2013")[0].strip()) if "\u2013" in g else 0:
                        pass
                nums.update(int(x) for x in re.split(r"[,\u2013-]", g) if x.strip().isdigit())
            if nums and num not in nums:
                problems.append(
                    f"锚点「{keyword}」附近引用了 {sorted(nums)}，但该话题期望的 "
                    f"[{num}]（{expected_surname}）不在其中")

    print()
    if problems:
        print(f"发现 {len(problems)} 项需人工确认：")
        for p in problems:
            print("   -", p)
        return 1
    print("双向完整性与锚点语义检查均通过 ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
