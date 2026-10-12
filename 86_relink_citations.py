"""按「正文语义」重排引用号并重建文献表（最终版）。

历史教训（本文件的存在理由）
--------------------------
本会话的引用重排经历过三次失败，每次都在此留痕：

1. `scripts/82` 只改写正文引用号、**没有同步改文献表** → 正文 `[8]` 指 van den Berg，
   表中 `[8]` 却是 Geirhos。编号本身完全合法，**肉眼看不出错位**。
2. `scripts/83` 依据编号判断"表中未被引用"，把 Wagner 条目**误删**。
3. `scripts/86` 初版试图用"锚点短语"自动定位引用括号，但锚点是我凭记忆写的措辞，
   与文件实际文本不符，反复失配。

最终方案：**把 `CURRENT_TO_FINAL` 写死为人工核对过的映射**，依据是从
`Introduction_Discussion_可直接引用.md` 中**逐条抽取的原文语境**（见下方注释），
不再依赖任何自动定位。脚本只做三件事：按映射改写正文引用、按 FINAL_ENTRIES 重建表、
并做双向完整性自检。

用法：
    python scripts/86_relink_citations.py --dry-run
    python scripts/86_relink_citations.py
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANU = ROOT / "docs" / "manuscript"
DOC = MANU / "Introduction_Discussion_可直接引用.md"
METHODS = MANU / "Methods_可直接引用.md"

# 旧号 -> 新号。依据是逐条抽取的**原文语境**（括号内为核对时看到的语境要点）：
#   [1] "excluding it"                        -> Bray GLOBOCAN             = 1
#   [2] "rather than confirming it"           -> Mohammadi                = 2
#   [3] "AUC can remain stable"               -> Thiringer                = 3
#   [4-7] "whole-slide image (WSI) analysis"  -> Chen/Lu/Xu/Filiot        = 4-7（不变）
#   [8] "further model ranking"               -> van den Berg             = 8（不变）
#   [9] "correlate with the label"            -> Geirhos                  = 9（不变）
#   [10] "metal tokens rather than pathology" -> Zech                    = 10（不变）
#   [11,12] "benefit is task-dependent"       -> Tellez/Jahanifar         = 11,12（不变）
#   [13] "questioned its practical value"     -> Boschman                 = 13（不变）
#   [14,15] "rank models"/"common protocol"   -> Campanella/Neidlinger    = 14,15（不变）
#   [3,12,13] "scanner or centre"             -> Thiringer/Boschman/Camp. （不变）
#   [16,17] "surveyed and benchmarked"        -> Zamanitajeddin/Aubreville= 16,17（不变）
#   [18] "mitosis detection across scanners"  -> **错位**：该语境是 MIDOG(Aubreville)，
#                                                却占着 ENDO-AID 的位置        -> 并入 17
#   [19] "endoscopic biopsies"                -> Wagner                   = 19（不变）
#   Methods [17] "Reinhard colour normalisation" -> **错位**（原指 Aubreville） -> Reinhard = 18
#   Methods [20] "attention-based MIL (ABMIL)"   -> Ilse                  = 20（不变）
#   Methods [21] "clustering-constrained"        -> CLAM                  = 21（不变）
CURRENT_TO_FINAL: dict[int, int] = {
    1: 1, 2: 2, 3: 3,
    4: 4, 5: 5, 6: 6, 7: 7,
    8: 8, 9: 9, 10: 10,
    11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17,
    18: 17,          # [18] 的语境是 MIDOG（Aubreville），归并到 17
    19: 19,          # Wagner
}

# Methods 内的错位修复：Reinhard 目前挂着 [17]（= Aubreville），应改为 [18]
METHODS_FIXES: list[tuple[str, str]] = [
    ("Reinhard colour normalisation [17]", "Reinhard colour normalisation [18]"),
]

FINAL_ENTRIES: list[tuple[str, str]] = [
    ("Bray F, Laversanne M, Sung H, Ferlay J, et al. Global cancer statistics 2022: GLOBOCAN "
     "estimates of incidence and mortality worldwide for 36 cancers in 185 countries. "
     "*CA Cancer J Clin* 2024;74(3):229-263. doi:10.3322/caac.21834", "✅ Crossref 直查 DOI"),
    ("Mohammadi M, Fell C, Bell S, et al. Endometrial whole-slide images dataset for detection "
     "of malignancy in endometrial biopsies. *GigaScience* 2025;14:giaf147. "
     "doi:10.1093/gigascience/giaf147", "既有综述 `[VERIFIED]`"),
    ("Thiringer E, Gustafsson FK, Eriksson KL, Rantalainen M. Scanner-induced domain shifts "
     "undermine the robustness of pathology foundation models. arXiv:2601.04163. 2026",
     "既有综述（预印本）"),
    ("Chen RJ, Ding T, Lu MY, et al. Towards a general-purpose foundation model for "
     "computational pathology. *Nat Med* 2024;30(3):850-862. doi:10.1038/s41591-024-02857-3",
     "既有综述 `[VERIFIED]`"),
    ("Lu MY, Chen B, Williamson DFK, et al. A visual-language foundation model for computational "
     "pathology. *Nat Med* 2024;30(3):863-874. doi:10.1038/s41591-024-02856-4",
     "既有综述 `[VERIFIED]`"),
    ("Xu H, Usuyama N, Bagga J, et al. A whole-slide foundation model for digital pathology from "
     "real-world data. *Nature* 2024;630:181-188. doi:10.1038/s41586-024-07441-w",
     "既有综述 `[VERIFIED]`"),
    ("Filiot A, Ghermi R, Olivier A, et al. Scaling self-supervised learning for histopathology. "
     "arXiv:2409.09173. 2024（**Phikon-v2，本文主编码器**）", "既有综述（预印本）"),
    ("van den Berg NJ, Volinsky-Fremond S, Wolf JB, et al. Assessing foundation models for "
     "computational pathology in endometrial cancer. *Mod Pathol* 2026;39(8):101070. "
     "doi:10.1016/j.modpat.2026.101070", "既有综述 `[VERIFIED]`"),
    ("Geirhos R, Jacobsen J-H, Michaelis C, Zemel R, Brendel W, Bethge M, Wichmann FA. Shortcut "
     "learning in deep neural networks. *Nat Mach Intell* 2020;2(11):665-673. "
     "doi:10.1038/s42256-020-00257-z", "✅ Crossref 直查 DOI"),
    ("Zech JR, Badgeley MA, Liu M, Costa AB, Titano JJ, Oermann EK. Variable generalization "
     "performance of a deep learning model to detect pneumonia in chest radiographs: a "
     "cross-sectional study. *PLoS Med* 2018;15:e1002683. doi:10.1371/journal.pmed.1002683. "
     "PMID 30399157", "✅ Europe PMC 检索核实"),
    ("Tellez D, Litjens G, Bándi P, et al. Quantifying the effects of data augmentation and stain "
     "color normalization in convolutional neural networks for computational pathology. "
     "*Med Image Anal* 2019;58:101544. doi:10.1016/j.media.2019.101544. PMID 31466046",
     "既有综述 `[VERIFIED]`"),
    ("Jahanifar M, Raza SEA, Xu K, et al. Domain generalisation in computational pathology: a "
     "survey and guideline. arXiv:2310.19656. 2023", "既有综述 `[VERIFIED]`"),
    ("Boschman J, Farahani H, Darbandsari A, et al. The utility of color normalization for "
     "AI-based diagnosis of hematoxylin and eosin-stained pathology images. *J Pathol* "
     "2022;256(1):15-24. doi:10.1002/path.5797. PMID 34543435", "既有综述 `[VERIFIED]`"),
    ("Campanella G, et al. A clinical benchmark of public self-supervised pathology foundation "
     "models. *Nat Commun* 2025;16:3640. doi:10.1038/s41467-025-58796-1", "既有综述 `[VERIFIED]`"),
    ("Neidlinger P, El Nahhas OSM, Muti HS, et al. Benchmarking foundation models as feature "
     "extractors for weakly supervised computational pathology. *Nat Biomed Eng* 2025. "
     "doi:10.1038/s41551-025-01516-3", "既有综述 `[VERIFIED]`"),
    ("Zamanitajeddin N, et al. Towards robust foundation models for digital pathology. "
     "arXiv:2409.17063（30 种 DG 算法 benchmark；HISTOPANTUM）", "既有综述 `[VERIFIED]`"),
    ("Aubreville M, Stathonikos N, Bertram CA, et al. Mitosis domain generalization in "
     "histopathology images — the MIDOG challenge. *Med Image Anal* 2023;84:102699. "
     "doi:10.1016/j.media.2022.102699", "既有综述 `[VERIFIED]`"),
    ("Reinhard E, Adhikhmin M, Gooch B, Shirley P. Color transfer between images. "
     "*IEEE Comput Graph Appl* 2001;21(5):34-41. doi:10.1109/38.946629",
     "✅ Crossref 检索核实"),
    ("Wagner SJ, Reisenbüchler D, West NP, et al. Transformer-based biomarker prediction from "
     "colorectal cancer histology. *Cancer Cell* 2023;41(9):1650-1661. "
     "doi:10.1016/j.ccell.2023.08.002", "既有综述 `[VERIFIED]`"),
    ("Ilse M, Tomczak JM, Welling M. Attention-based deep multiple instance learning. *ICML* 2018",
     "既有综述（方法引用）"),
    ("Lu MY, Williamson DFK, Chen TY, et al. Data-efficient and weakly supervised computational "
     "pathology on whole-slide images. *Nat Biomed Eng* 2021;5:555-570（CLAM）",
     "既有综述（方法引用）"),
]

DATA_ENTRIES: list[tuple[str, str]] = [
    ("Gelton T, Vermorgen S, Ciompi F. ENDO-AID challenge dataset. Zenodo. "
     "doi:10.5281/zenodo.7372187", "既有综述（数据）"),
]

CITE_RE = re.compile(r"\[(\d+(?:\s*[,\u2013-]\s*\d+)*)\]")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    doc = DOC.read_text(encoding="utf-8")
    meth = METHODS.read_text(encoding="utf-8")

    for old, new in METHODS_FIXES:
        if old in meth:
            meth = meth.replace(old, new)
            print(f"  Methods 修复：{old.split()[-1]} -> {new.split()[-1]}")
        elif new in meth:
            print(f"  Methods 已是目标状态：{new.split()[-1]}")
        else:
            print(f"  [WARN] Methods 未找到待修文本：{old[:50]}")

    def repl(m: re.Match) -> str:
        g = m.group(1)
        nums = [int(x) for x in re.split(r"[,\u2013-]", g) if x.strip().isdigit()]
        if not nums:
            return m.group(0)
        is_range = ("\u2013" in g or "-" in g) and len(nums) == 2
        if is_range:
            span = list(range(min(nums), max(nums) + 1))
            if not all(n in CURRENT_TO_FINAL for n in span):
                return m.group(0)
            new = sorted({CURRENT_TO_FINAL[n] for n in span})
            if len(new) >= 3 and new == list(range(new[0], new[-1] + 1)):
                return f"[{new[0]}\u2013{new[-1]}]"
            return "[" + ",".join(str(x) for x in new) + "]"
        if all(n in CURRENT_TO_FINAL for n in nums):
            return "[" + ",".join(str(CURRENT_TO_FINAL[n]) for n in nums) + "]"
        return m.group(0)

    new_doc = CITE_RE.sub(repl, doc)

    tbl = ["| # | 文献 | 核实方式 |", "| --- | --- | --- |"]
    for i, (body, how) in enumerate(FINAL_ENTRIES, 1):
        tbl.append(f"| {i} | {body} | {how} |")
    tbl += ["", "**数据引用（单独编号，不占文献序列）**", "",
            "| # | 数据 | 说明 |", "| --- | --- | --- |"]
    for i, (body, how) in enumerate(DATA_ENTRIES, 1):
        tbl.append(f"| D{i} | {body} | {how} |")
    new_table = "\n".join(tbl)

    tl = new_doc.splitlines()
    try:
        start = next(i for i, ln in enumerate(tl) if ln.startswith("| # | 文献"))
    except StopIteration:
        print("[ERROR] 未找到文献表", file=sys.stderr)
        return 2
    end = start
    while end < len(tl) and (tl[end].startswith("|") or tl[end].strip() == ""
                             or tl[end].startswith("**数据引用")):
        end += 1
    new_doc = new_doc.replace("\n".join(tl[start:end]), new_table, 1)

    cited: set[int] = set()
    for text in (new_doc, meth):
        body = re.sub(r"^>.*$", "", text, flags=re.M)
        for m in CITE_RE.finditer(body):
            g = m.group(1)
            if "\u2013" in g or "-" in g:
                p = [x.strip() for x in re.split(r"[,\u2013-]", g)]
                if len(p) == 2 and all(x.isdigit() for x in p):
                    cited.update(range(int(p[0]), int(p[1]) + 1))
                    continue
            cited.update(int(x) for x in g.split(",") if x.strip().isdigit())
    n = len(FINAL_ENTRIES)
    over = sorted(x for x in cited if x > n)
    unused = sorted(set(range(1, n + 1)) - cited)
    if over:
        print(f"[WARN] 正文引用超出表范围：{over}")
    if unused:
        print(f"[WARN] 表中未被引用：{unused}")

    if args.dry_run:
        print(f"\n[dry] 新表 {n} 条文献 + {len(DATA_ENTRIES)} 条数据引用")
        print(f"      正文引用编号：{sorted(cited)}")
        return 0

    DOC.write_text(new_doc, encoding="utf-8")
    METHODS.write_text(meth, encoding="utf-8")
    for p in (DOC, METHODS):
        p.read_text(encoding="utf-8")
    print(f"\n已重建文献表：{n} 条文献 + {len(DATA_ENTRIES)} 条数据引用；UTF-8 校验通过")
    print(f"正文引用编号：{sorted(cited)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
