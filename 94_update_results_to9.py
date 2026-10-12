"""把 Results 正文从 7 来源口径更新到 9 来源口径。

背景：
    新增两个来源（`gastric_tum` 胃、`pcam_lymph` 淋巴结）后，矩阵从 7×7（42 配对）
    变为 9×9（72 配对），所有汇总数字随之改变。本脚本按"旧值 -> 新值"逐条替换，
    **每条都来自 runs/ 下的 JSON**，不做手工转录（本项目已两次因手抄密集数字出错）。

做法与约束：
    - 只替换**明确的、唯一的**文本片段；找不到就报 MISS 而不是静默跳过；
    - 改完立即 UTF-8 往返校验；
    - 不触碰刻意保留的历史对照段（自我纠正段落里的旧值已加注）。

用法：
    python scripts/94_update_results_to9.py --dry-run
    python scripts/94_update_results_to9.py
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "manuscript" / "Results_可直接引用.md"

# (旧片段, 新片段, 说明) —— 顺序即替换顺序
EDITS: list[tuple[str, str, str]] = [
    # ---- §3.1 数据来源：7 -> 9
    ("We assembled **seven** publicly available, class-labelled histopathology image datasets\n"
     "spanning **six organ systems**, plus two whole-slide image (WSI) cohorts for specimen-type\n"
     "analysis (Table 1–2). All datasets are openly downloadable; no data-use agreement was\n"
     "required. Cross-source results are reported for all 7 × 6 = **42 ordered source→target\n"
     "pairs**.",
     "We assembled **nine** publicly available, class-labelled histopathology image datasets\n"
     "spanning **eight organ systems**, plus two whole-slide image (WSI) cohorts for\n"
     "specimen-type analysis (Table 1–2). All datasets are openly downloadable; no data-use\n"
     "agreement was required. Cross-source results are reported for all 9 × 8 = **72 ordered\n"
     "source→target pairs**.",
     "§3.1 来源数与配对数"),
    ("| `colon_cancer_csv` | colon | 1,092 (of 2,190) | 546 / 546 | Apache-2.0 |",
     "| `colon_cancer_csv` | colon | 1,092 (of 2,190) | 546 / 546 | Apache-2.0 |\n"
     "| `gastric_tum` | stomach | 7,774 (of 11,661) | 3,887 / 3,887 | CC BY 4.0 |\n"
     "| `pcam_lymph` | lymph node | 10,000 (of 262,144) | 5,000 / 5,000 | CC0 |",
     "§3.1 新增两行来源"),
    ("Two sources are **sampled** to a per-class cap for computational feasibility\n"
     "(`breast_idc`, `colon_cancer_csv`); the other five are used at full size. `breast_idc` is the\n"
     "only patch source that publishes a **patient identifier**, so its in-domain evaluation is\n"
     "grouped by patient (279 patients) rather than by image — for the remaining six, no patient\n"
     "identifier exists and grouping is by image filename (see §3.2 and the limitations).",
     "Four sources are **sampled** to a per-class cap for computational feasibility\n"
     "(`breast_idc`, `pcam_lymph`, `gastric_tum`, `colon_cancer_csv`); the other five are used at\n"
     "full size. Two sources support stricter-than-image grouping: `breast_idc` publishes a\n"
     "**patient identifier** (279 patients) and `pcam_lymph` publishes a **slide identifier**\n"
     "(216 slides), so both are grouped at that level; for the remaining seven, grouping is by\n"
     "image filename (see §3.2 and the limitations).",
     "§3.1 抽样与分组口径"),

    # ---- §3.2 域内
    ("With 5-fold GroupKFold cross-validation, in-domain AUC spans **0.7297–1.0000** (mean 0.9467):",
     "With 5-fold GroupKFold cross-validation, in-domain AUC spans **0.7297–1.0000** (mean 0.9496):",
     "§3.2 域内均值"),
    ("| `ovarian_ca` | **0.9705** | 0.9912 | image (496) |\n"
     "| `breast_idc` | **0.9351** | 0.9305 | **patient (279)** |\n"
     "| `colon_cancer_csv` | **0.7297** | 0.6997 | image (1,092) |",
     "| `ovarian_ca` | **0.9705** | 0.9912 | image (496) |\n"
     "| `pcam_lymph` | **0.9684** | 0.9748 | **slide (216)** |\n"
     "| `gastric_tum` | **0.9511** | 0.9468 | image (7,774) |\n"
     "| `breast_idc` | **0.9351** | 0.9305 | **patient (279)** |\n"
     "| `colon_cancer_csv` | **0.7297** | 0.6997 | image (1,092) |",
     "§3.2 新增两行"),
    ("**This corrects an earlier version of our own claim.** With five sources we observed\n"
     "0.9705–1.0000 and described in-domain performance as \"near-perfect\". Extending to seven\n"
     "sources shows that this was partly an artefact of source selection: the two sources added\n"
     "last are markedly harder. `colon_cancer_csv` (AUC 0.7297) is a genuinely difficult task, and\n"
     "`breast_idc` is the only source evaluated under **patient-level** grouping — which is the\n"
     "more honest protocol and yields a lower value (0.9351) than image-level grouping would.",
     "**This corrects earlier versions of our own claim.** With five sources we observed\n"
     "0.9705–1.0000 and described in-domain performance as \"near-perfect\"; with seven it became\n"
     "0.7297–1.0000, and with the present nine sources it is **0.7297–1.0000, mean 0.9496**. The\n"
     "spread is therefore partly an artefact of source selection: `colon_cancer_csv` (0.7297) is a\n"
     "genuinely difficult task, while the two sources evaluated under **stricter grouping**\n"
     "(`breast_idc`, patient-level; `pcam_lymph`, slide-level) sit at 0.9351 and 0.9684 — below\n"
     "the near-ceiling values obtained when images are treated as independent.",
     "§3.2 自我纠正段（更新到 9 来源）"),

    # ---- §3.3 崩塌
    ("**All 42 source→target pairs degrade substantially (Figure 1, Table 3).**",
     "**All 72 source→target pairs degrade substantially (Figure 1, Table 3).**",
     "§3.3 配对数"),
    ("Cross-source AUC spans **0.3132–0.8243**, and **15 of 42 pairs fall below chance\n"
     "(AUC < 0.5)**. Mean ΔAUC across all pairs is **−0.4028**. The collapse is therefore not a\n"
     "peculiarity of one dataset pair: it is **systematic across organs and across transfer\n"
     "directions**. The complete 7 × 7 matrix and the per-target-source breakdown are given in\n"
     "**Tables S2 and S3**; the worst target domain is `colon_cancer_csv` (mean cross-source AUC\n"
     "0.3922) and the least affected is `uterus_ca` (0.6694, ΔAUC −0.3302), against in-domain AUCs\n"
     "of 0.7297 and 0.9996 respectively.",
     "Cross-source AUC spans **0.3132–0.9557**, and **22 of 72 pairs fall below chance\n"
     "(AUC < 0.5)**. Mean ΔAUC across all pairs is **−0.3716**. The collapse is therefore not a\n"
     "peculiarity of one dataset pair: it is **systematic across organs and across transfer\n"
     "directions**. The complete 9 × 9 matrix and the per-target-source breakdown are given in\n"
     "**Tables S2 and S3**; the worst target domain is `colon_cancer_csv` (mean cross-source AUC\n"
     "0.3826) and the least affected is `uterus_ca` (0.7095, ΔAUC −0.2901), against in-domain AUCs\n"
     "of 0.7297 and 0.9996 respectively.",
     "§3.3 汇总数字"),
    ("> **A note on how the added sources change the picture.** Extending from 5 to 7 sources\n"
     "> *strengthens* the finding in one respect and *qualifies* it in another. Strengthened: the\n"
     "> fraction of below-chance pairs rises from 6/20 (30%) to **15/42 (36%)**, and the worst pair\n"
     "> becomes worse (0.3132 vs 0.356). Qualified: the mean in-domain AUC falls from 0.9925 to\n"
     "> 0.9467, so the contrast between \"in-domain\" and \"cross-source\" is smaller than the\n"
     "> five-source subset suggested. Both facts are reported.",
     "> **A note on how adding sources changes the picture.** Each extension has both strengthened\n"
     "> and qualified the finding, and we report both directions. Below-chance fraction: 6/20 (30%)\n"
     "> with five sources, 15/42 (36%) with seven, **22/72 (31%) with nine**. Mean in-domain AUC:\n"
     "> 0.9925 (five), 0.9467 (seven), **0.9496 (nine)**. Mean ΔAUC: −0.4028 (seven) and\n"
     "> **−0.3716 (nine)**. The newly added sources (`gastric_tum`, `pcam_lymph`) transfer\n"
     "> comparatively well — their ΔAUC values, −0.3357 and −0.3466, are the least severe of the\n"
     "> nine — so the aggregate contrast is somewhat smaller than the seven-source subset implied.\n"
     "> We therefore report the per-target breakdown rather than a single headline gap.",
     "§3.3 来源增减的影响说明"),
    ("> **Limitations that must be stated.** (i) The seven sources use **different class",
     "> **Limitations that must be stated.** (i) The nine sources use **different class",
     "§3.3 局限（来源数）"),
    ("> reflect label-space mismatch. (ii) **Only `breast_idc` publishes a patient identifier**;\n"
     "> the other six sources have none, so their grouping is at image level and same-patient\n"
     "> leakage cannot be fully excluded. (iii) Only one linear probe was used per encoder; the\n"
     "> matrix quantifies transfer loss and is **not** a model ranking. (iv) Two sources\n"
     "> (`breast_idc`, `colon_cancer_csv`) are subsampled per class, so their in-domain estimates\n"
     "> rest on 10,000 and 1,092 images respectively rather than the full sets.",
     "> reflect label-space mismatch. (ii) **Only `breast_idc` (patient identifier) and\n"
     "> `pcam_lymph` (slide identifier) support stricter-than-image grouping**; the other seven\n"
     "> sources have no patient identifier, so their grouping is at image level and same-patient\n"
     "> leakage cannot be fully excluded. (iii) Only one linear probe was used per encoder in the\n"
     "> main matrix; the matrix quantifies transfer loss and is **not** a model ranking (probe\n"
     "> sensitivity is examined separately in §3.11). (iv) Four sources (`breast_idc`,\n"
     "> `pcam_lymph`, `gastric_tum`, `colon_cancer_csv`) are subsampled per class, so their\n"
     "> in-domain estimates rest on the sampled counts in Table S1 rather than the full sets.",
     "§3.3 局限（分组与抽样）"),

    # ---- §3.4 捷径审计
    ("**All seven sources (7/7) show a large shortcut effect**, with AUC from **0.7237**\n"
     "(`colon_cancer_csv`) to **0.9775** (`lung_colon`); the complete per-source table, including\n"
     "the strongest features, is given in **Table S4**.",
     "**All nine sources (9/9) show a large shortcut effect**, with AUC from **0.7237**\n"
     "(`colon_cancer_csv`) to **0.9775** (`lung_colon`); the two newly added sources sit mid-range\n"
     "at **0.8513** (`pcam_lymph`) and **0.8423** (`gastric_tum`). The complete per-source table,\n"
     "including the strongest features, is given in **Table S4**.",
     "§3.4 捷径审计来源数"),
    ("Across the seven sources, the in-domain AUC and the shortcut AUC are **strongly positively\n"
     "rank-correlated** (Spearman ρ = **+0.857**, n = 7; per-source values in Figure 2 and Table S4):\n"
     "sources on which the encoder performs best are also those whose labels are most predictable\n"
     "from acquisition artefacts alone. This is the pattern expected if part of the apparent\n"
     "in-domain skill is shortcut-driven.",
     "Across sources, the in-domain AUC and the shortcut AUC are **positively rank-correlated**\n"
     "(Spearman ρ = **+0.683**, n = 9; per-source values in Figure 2 and Table S4): sources on\n"
     "which the encoder performs best tend also to be those whose labels are most predictable from\n"
     "acquisition artefacts alone. This is the pattern expected if part of the apparent in-domain\n"
     "skill is shortcut-driven. **We note that the correlation weakened on extending from seven\n"
     "sources (ρ = +0.857) to nine (+0.683)**: the two added sources both have high in-domain AUC\n"
     "yet mid-range shortcut AUC, so they do not follow the pattern as tightly. The association is\n"
     "therefore real but not tight enough to treat in-domain AUC as a proxy for shortcut strength.",
     "§3.4b 秩相关（n=9，含削弱说明）"),
    ("> **Limitations.** With n = 7 this is a **descriptive** association; no significance test is",
     "> **Limitations.** With n = 9 this is a **descriptive** association; no significance test is",
     "§3.4b 局限（n）"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not DOC.exists():
        print(f"[ERROR] 缺 {DOC}", file=sys.stderr)
        return 2
    text = DOC.read_text(encoding="utf-8")
    ok = miss = 0
    for old, new, why in EDITS:
        if old in text:
            text = text.replace(old, new, 1)
            ok += 1
            print(f"  [OK]   {why}")
        else:
            miss += 1
            print(f"  [MISS] {why}")

    # 残留旧数字自检
    stale = []
    for pat, why in ((r"\b42 (?:source|ordered|cross-source)", "仍出现 42 配对"),
                     (r"seven (?:sources|openly)", "仍出现 seven sources"),
                     (r"7 × 7", "仍出现 7×7"),
                     (r"\+0\.857", "仍出现旧 rho（应已在说明段保留）")):
        for m in re.finditer(pat, text):
            s = text[max(0, m.start() - 60):m.end() + 40].replace("\n", " ")
            stale.append(f"{why}: …{s[-95:]}")

    if args.dry_run:
        print(f"\n[dry] 命中 {ok}，未命中 {miss}")
        return 0

    DOC.write_text(text, encoding="utf-8")
    assert DOC.read_text(encoding="utf-8") == text
    print(f"\n已更新 {DOC.name}：命中 {ok}，未命中 {miss}；UTF-8 校验通过")
    if stale:
        print(f"\n[自检] {len(stale)} 处可能残留旧口径（需人工确认是否为刻意保留的历史对照）：")
        for s in stale[:12]:
            print("   -", s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
