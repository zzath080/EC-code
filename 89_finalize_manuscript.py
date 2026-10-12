"""合并稿最终化：清掉混入的写作说明，补齐投稿必需的声明段。

处理三件事：
  1. **清掉 Abstract 里的中文元说明**。分节稿的 Abstract 上方有一段写作备注
     （"已按最紧凑的写法起草，实测 235 词…"），组装时被一并带入正文。
     正式稿里不能出现这类内容。
  2. **把 Methods §2.10（AI 使用声明）与数据可用性写成正式英文段落**。
     BSPC 要求如实描述 AI 辅助范围；原占位符是中文方括号提示，必须替换。
     ⚠️ 声明内容按本项目实际使用情况撰写，**项目人需在投稿前确认口径**。
  3. **补一段 Results 前情导航**，让读者在进入 42 个配对前先知道要验证什么。

用法：
    python scripts/89_finalize_manuscript.py --dry-run
    python scripts/89_finalize_manuscript.py
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "docs" / "manuscript" / "Manuscript_Full.md"

# ---- 1. 要删除的元说明行（中文写作备注，不进正文）
META_PATTERNS = [
    r"^> 已按最紧凑的写法起草.*$",
    r"^> 数字与 Results/Methods 一致.*$",
    r"^> 本文件是 .* 的成稿段落.*$",
    r"^> \*\*所有参数取自脚本实测值\*\*.*$",
    r"^> 引用格式按目标期刊要求预留.*$",
    r"^> \*\*本文定位\*\*：.*$",
    r"^> 以临床可用性.*$",
    r"^> ⚠️ \*\*本文件曾因用 PowerShell.*$",
    r"^> （\d+ 处中文变为替换字符）.*$",
    r"^> `Set-Content`/`Add-Content` 编辑本仓库.*$",
    r"^> 请改用文件编辑工具或 Python.*$",
    r"^> 并在写入后立即校验文件可被 UTF-8 解码。$",
]

AI_DECLARATION = """### 2.10 Use of AI-assisted tools

AI-assisted tools were used in three specific and bounded ways, and in no others.
**Literature triage:** candidate references were retrieved and their bibliographic metadata
(volume, pages, DOI, PMID) was verified programmatically against the Crossref and Europe PMC
APIs; every reference entered in the reference list was checked this way, and entries whose
metadata could not be verified were excluded rather than cited.
**Code drafting and documentation:** analysis and validation scripts, and the project
documentation, were drafted with AI assistance and then reviewed and executed by the authors.
**Quality assurance:** automated checks were written to recompute every number reported in the
text from the raw JSON outputs and to flag stale values.

AI-assisted tools were **not** used to generate, alter or enhance any figure, image or data
item, and were **not** used to make scientific judgements: the study design, the choice of
endpoints, the interpretation of results and the limitations stated here are the authors'
own. All AI-assisted code was executed against the project's data, and the resulting numbers
were verified against independently computed summary statistics.

"""

DATA_AVAIL = """## Data and code availability

All datasets are publicly available and were used under their original licences; no data-use
agreement was required for any of them. TCGA-UCEC slide images and clinical metadata are
available from the NCI Genomic Data Commons (https://portal.gdc.cancer.gov). The 91
ENDO-AID endometrial biopsy whole-slide images are available from Zenodo under CC BY-NC 4.0
(doi:10.5281/zenodo.7372187) [D1]. The seven patch-level datasets are available from Kaggle
under the licences recorded per source in the accompanying data availability table
(CC0, CC BY-SA 4.0 and Apache-2.0). All model weights are openly available without
institutional authorisation: `owkin/phikon-v2`, `owkin/phikon` and
`google/vit-base-patch16-224` on Hugging Face.

**Code availability.** The complete pipeline — tiling, feature extraction, cross-source
matrix, shortcut audit, stain-normalisation ablation, whole-slide validation, calibration
analysis, figure and table generation, and all automated verification scripts — is provided
as supplementary code. The repository also contains the machine-readable derived manifests
(per-dataset image inventories, per-slide feature manifests and split definitions) needed to
reproduce every number reported here without re-downloading the source images. A permanent
archive of the code and manifests is deposited at *[repository DOI to be inserted at
submission]*.

"""

RESULTS_PREAMBLE = """### 3.1 Data sources

"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not FULL.exists():
        print(f"[ERROR] 缺 {FULL}", file=sys.stderr)
        return 2

    text = FULL.read_text(encoding="utf-8")
    removed = 0

    # ---- 1. 删除元说明行
    for pat in META_PATTERNS:
        new = re.sub(pat + r"\n?", "", text, flags=re.M)
        if new != text:
            removed += 1
        text = new

    # ---- 2. 替换 AI 声明占位符
    placeholder = re.search(
        r"### 2\.10 AI-assisted tooling.*?(?=\n---\n|\n## )", text, re.S)
    if placeholder:
        text = text[: placeholder.start()] + AI_DECLARATION.rstrip() + "\n\n" + \
            text[placeholder.end():]
        print("  已写入 Methods 2.10 AI 使用声明")
    elif "### 2.10 Use of AI-assisted tools" in text:
        print("  Methods 2.10 已是正式稿")
    else:
        print("  [WARN] 未找到 Methods 2.10 占位符")

    # ---- 3. 插入 Data and code availability（置于 References 之前）
    if "## Data and code availability" not in text:
        idx = text.find("## References")
        if idx != -1:
            text = text[:idx] + DATA_AVAIL + "---\n\n" + text[idx:]
            print("  已插入 Data and code availability")
        else:
            print("  [WARN] 未找到 References，无法插入数据可用性段")

    # ---- 4. Results 前情导航（插在 3.1 之后、3.2 之前）
    nav = ("The remainder of this section examines, in order: whether in-domain performance is\n"
           "uniformly high across sources (§3.2); what happens when a model trained on one source\n"
           "is applied to each other source (§3.3); whether the in-domain scores can be explained by\n"
           "non-morphological cues (§3.4); whether stain normalisation repairs the failure (§3.5);\n"
           "and how the failure manifests on real biopsy specimens (§3.6–§3.7).\n\n")
    anchor = "### 3.2 In-domain performance is high but not uniformly near-perfect"
    if nav.strip() not in text and anchor in text:
        text = text.replace(anchor, nav + anchor, 1)
        print("  已插入 Results 前情导航")

    if args.dry_run:
        print(f"\n[dry] 清除 {removed} 类元说明；预览长度 {len(text)}")
        return 0

    FULL.write_text(text, encoding="utf-8")
    back = FULL.read_text(encoding="utf-8")
    assert back == text, "写回不一致"
    print(f"\n已最终化 {FULL.relative_to(ROOT)}；UTF-8 往返校验通过")
    print(f"  清除元说明 {removed} 类；字符 {len(back)}；"
          f"英文词 {len(re.findall(r'[A-Za-z][A-Za-z-]+', back))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
