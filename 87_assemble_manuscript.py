"""把分节稿件合并成一份可直接投稿的完整稿（Abstract -> References）。

组装规则：
    顺序 = Title -> Abstract -> Keywords -> Highlights -> 1. Introduction ->
    2. Materials and Methods -> 3. Results -> 4. Discussion -> 5. Conclusion ->
    Data and Code Availability -> References -> 图注/表注 -> 补充材料索引。

    各分节正文从三份已成稿文件中**按小节抽取**，而不是手工复制，避免出现
    "改了分节稿、忘了改合并稿"之类的漂移。

刻意剔除的内容（属于写作辅助，不进正文）：
    - 「已核实文献」表的核实方式列之外的中文说明；文献表本身保留（转为英文引用格式）；
    - 「词数说明」「投稿前仍需处理」等元信息；
    - 分节稿开头的说明性引用块。

用法：
    python scripts/87_assemble_manuscript.py
输出：
    docs/manuscript/Manuscript_Full.md
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANU = ROOT / "docs" / "manuscript"
OUT = MANU / "Manuscript_Full.md"

INTRO = MANU / "Introduction_Discussion_可直接引用.md"
METH = MANU / "Methods_可直接引用.md"
RES = MANU / "Results_可直接引用.md"

TITLE = ("Cross-source fragility of pathology foundation model embeddings: "
         "a calibration- and shortcut-aware benchmark across organ and specimen types")

KEYWORDS = ("endometrial cancer; whole-slide image; foundation model; domain shift; "
            "calibration; benchmarking")

HIGHLIGHTS = [
    "Cross-source AUC collapsed by 0.35-0.42 in all three frozen encoders tested",
    "Non-morphological image statistics alone reached AUC 0.80-0.98",
    "Stain normalisation did not repair the collapse; it was encoder-specific",
    "Biopsy sensitivity ranged 0.02-0.95 across encoders at one 0.5 threshold",
    "AUC did not predict usability: calibration and threshold behaviour must be reported",
]


def strip_meta(text: str) -> str:
    """去掉分节稿开头的说明性引用块与文末写作辅助段。"""
    lines = text.splitlines()
    out, i = [], 0
    # 保留首个 H1（会有选择地使用），跳过其后的连续引用块
    while i < len(lines):
        if lines[i].startswith(">"):
            i += 1
            continue
        break
    out.extend(lines[i:])
    text = "\n".join(out)
    # 截断写作辅助章节
    for marker in ("## 3. 已核实文献", "## 3. 已核实文献（", "## 4. 词数说明",
                   "## 附：", "## 附录", "## 0. Abstract"):
        idx = text.find(marker)
        if idx != -1 and marker != "## 0. Abstract":
            text = text[:idx]
    return text.rstrip()


def extract_section(text: str, start_marker: str, stop_markers: list[str]) -> str:
    """抽取从 start_marker 起、到任一 stop_marker 止的内容。"""
    i = text.find(start_marker)
    if i == -1:
        return ""
    j = len(text)
    for sm in stop_markers:
        k = text.find(sm, i + len(start_marker))
        if k != -1:
            j = min(j, k)
    return text[i:j].rstrip()


def build_references() -> str:
    """从 Introduction_Discussion 的文献表转成英文引用列表。"""
    text = INTRO.read_text(encoding="utf-8")
    lines = text.splitlines()
    try:
        start = next(i for i, ln in enumerate(lines) if ln.startswith("| # | 文献"))
    except StopIteration:
        return "*(references unavailable)*"
    end = start
    while end < len(lines) and (lines[end].startswith("|") or lines[end].strip() == ""
                                or lines[end].startswith("**数据引用")):
        end += 1
    entries: list[tuple[str, str]] = []
    data: list[tuple[str, str]] = []
    in_data = False
    for ln in lines[start:end]:
        if ln.startswith("**数据引用"):
            in_data = True
            continue
        m = re.match(r"^\|\s*(\d+|D\d+)\s*\|\s*(.+?)\s*\|\s*(.*?)\s*\|\s*$", ln)
        if not m or m.group(1) == "#":
            continue
        (data if in_data else entries).append((m.group(1), m.group(2).strip()))

    out = ["## References", ""]
    for n, body in entries:
        # 去掉核实标记加粗，规范成编号引用格式
        b = body.replace("**", "").strip()
        out.append(f"{n}. {b}")
    if data:
        out += ["", "**Data references**", ""]
        for n, body in data:
            out.append(f"{n}. {body.replace('**', '').strip()}")
    # 去掉核实方式里的中文，仅保留条目本身（核实信息留在分节稿中）
    return "\n".join(out)


def main() -> int:
    for p in (INTRO, METH, RES):
        if not p.exists():
            print(f"[ERROR] 缺 {p}", file=sys.stderr)
            return 2

    intro_raw = INTRO.read_text(encoding="utf-8")
    meth_raw = METH.read_text(encoding="utf-8")
    res_raw = RES.read_text(encoding="utf-8")

    # Abstract：从 Introduction_Discussion 的 "## 0. Abstract" 抽到下一个 "---"
    m = re.search(r"## 0\. Abstract.*?\n(.*?)\n---\n", intro_raw, re.S)
    abstract = (m.group(1).strip() if m else "*(abstract unavailable)*")
    # 分节稿的 Abstract 上方带中文写作备注（"已按最紧凑的写法起草…"），
    # 组装时必须剔除，否则写作说明会被带进正文。
    abstract = "\n".join(ln for ln in abstract.splitlines()
                         if not ln.lstrip().startswith(">")).strip()

    # Introduction：跳过 "## 0. Abstract"，抽 "## 1. Introduction" 到 "## 2. Discussion"
    intro = extract_section(intro_raw, "## 1. Introduction", ["## 2. Discussion"])
    disc = extract_section(intro_raw, "## 2. Discussion", ["## 3. 已核实文献"])

    # 分节稿里 Discussion 编号是 "2."（因为它与 Introduction 同处一个文件）。
    # 合并后 Methods=2、Results=3，Discussion 必须是 **4**，Conclusion 必须是 **5**。
    disc = disc.replace("## 2. Discussion", "## 4. Discussion", 1)
    for i in range(1, 7):
        disc = disc.replace(f"### 2.{i} ", f"### 4.{i} ", 1)
    # 结论在分节稿里是 4.6；合并稿中提升为独立一章
    disc = disc.replace("### 4.6 Conclusion", "## 5. Conclusion", 1)

    methods = strip_meta(meth_raw)
    methods = extract_section(methods, "## 2. Materials and methods", ["## 附："])
    results = strip_meta(res_raw)
    results = extract_section(res_raw, "## 3.1 Data sources", ["## 附：", "## 附"])

    # 标题里的 markdown 强调标记在正式稿中会渲染成粗体，这里清掉以便排版统一
    results = results.replace("## 3.2 In-domain performance is high but **not uniformly "
                              "near-perfect**",
                              "## 3.2 In-domain performance is high but not uniformly near-perfect")
    # 分节稿里 Results 直接以 "## 3.1" 开头（缺父标题）。BSPC 要求章节层级完整，
    # 因此补 "## 3. Results"，并把 3.x 降为 "### 3.x"（3.4b/3.5b 保持三级）。
    results = "## 3. Results\n\n" + results
    results = re.sub(r"^## (3\.\d+ )", r"### \1", results, flags=re.M)

    parts = [
        f"# {TITLE}", "",
        "## Abstract", "", abstract, "",
        f"**Keywords:** {KEYWORDS}", "",
        "## Highlights", "",
    ]
    parts += [f"- {h}" for h in HIGHLIGHTS]
    parts += ["", "---", "", intro, "", "---", "", methods, "", "---", "", results,
              "", "---", "", disc, "", "---", "",
              build_references(), "", "---", "",
              "## Figure and table legends", "",
              "See `FIGURE_TABLE_LEGENDS.md` (Figures 1–5, Tables 3–8) and "
              "`../paper_tables/Table1_cohort_characteristics.md`, "
              "`../paper_tables/Table2_datasets.md` (Tables 1–2).", "",
              "## Supplementary material", "",
              "Tables S1–S8 are provided in `Supplementary_Material.md`.", ""]

    text = "\n".join(parts)
    OUT.write_text(text, encoding="utf-8")
    back = OUT.read_text(encoding="utf-8")
    assert back == text, "写回不一致"

    words = len(re.findall(r"[A-Za-z][A-Za-z'-]+", back))
    print(f"已生成 {OUT.relative_to(ROOT)}")
    print(f"  UTF-8 往返校验通过；字符 {len(back)}；英文词 {words}")
    for name, seg in (("Introduction", intro), ("Methods", methods),
                      ("Results", results), ("Discussion", disc)):
        print(f"  {name:12s} {len(re.findall(r'[A-Za-z][A-Za-z-]+', seg)):5d} 词")
    return 0


if __name__ == "__main__":
    sys.exit(main())
