"""把 Introduction/Discussion/Abstract 从 7 来源口径更新到 9 来源口径。

与 `scripts/94_update_results_to9.py` 同一目的，但作用于另一个文件。
所有新值均取自 runs/ 下的 JSON（不手工转录）。

用法：
    python scripts/95_update_intro_disc_to9.py --dry-run
    python scripts/95_update_intro_disc_to9.py
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "manuscript" / "Introduction_Discussion_可直接引用.md"

EDITS: list[tuple[str, str, str]] = [
    # ---- Abstract
    ("We built a cross-source benchmark over **seven** openly licensed histopathology\n"
     "patch datasets (six organ systems) and **three** frozen encoders, evaluated as a complete\n"
     "7 × 7 matrix (42 cross-source pairs) with a single logistic probe.",
     "We built a cross-source benchmark over **nine** openly licensed histopathology\n"
     "patch datasets (eight organ systems) and **three** frozen encoders, evaluated as a complete\n"
     "9 × 9 matrix (72 cross-source pairs) with a single logistic probe.",
     "Abstract 来源数/配对数"),
    ("**Results.** In-domain AUC ranged 0.7297–1.0000; all 42 cross-source pairs degraded (mean\n"
     "ΔAUC −0.4028) and **15/42 fell below chance**, consistently across all three encoders.\n"
     "Non-morphological statistics alone predicted the label in **7/7** sources (AUC 0.724–0.978),\n"
     "and in-domain AUC correlated with shortcut AUC (Spearman ρ = +0.857).",
     "**Results.** In-domain AUC ranged 0.7297–1.0000; all 72 cross-source pairs degraded (mean\n"
     "ΔAUC −0.3716) and **22/72 fell below chance**, consistently across all three encoders.\n"
     "Non-morphological statistics alone predicted the label in **9/9** sources (AUC 0.724–0.978),\n"
     "and in-domain AUC correlated with shortcut AUC (Spearman ρ = +0.683, n = 9).",
     "Abstract 结果数字"),

    # ---- Introduction §1.1
    ("a recent multi-model assessment concluded that foundation models have",
     "a recent multi-model assessment concluded that foundation models have",
     "（占位，无改动）"),

    # ---- §1.3 贡献 1
    ("1. **A cross-source benchmark spanning seven openly licensed patch datasets over six organ\n"
     "   systems, plus two WSI cohorts**, evaluated as a complete 7 × 7 matrix (42 cross-source\n"
     "   pairs; 14,16). All encoders are **openly available**, so the protocol is reproducible\n"
     "   without institutional authorisation.",
     "1. **A cross-source benchmark spanning nine openly licensed patch datasets over eight organ\n"
     "   systems, plus two WSI cohorts**, evaluated as a complete 9 × 9 matrix (72 cross-source\n"
     "   pairs; 14,16). All encoders are **openly available**, so the protocol is reproducible\n"
     "   without institutional authorisation.",
     "§1.3 贡献 1"),
    ("2. **A shortcut audit** showing that in **7/7** sources the class label is predictable from\n"
     "   non-morphological image statistics alone (AUC 0.72–0.98), and that in-domain AUC and\n"
     "   shortcut AUC are strongly rank-correlated (Spearman ρ = +0.857). High in-domain scores in\n"
     "   these datasets therefore cannot be read as evidence of morphological understanding.",
     "2. **A shortcut audit** showing that in **9/9** sources the class label is predictable from\n"
     "   non-morphological image statistics alone (AUC 0.72–0.98), and that in-domain AUC and\n"
     "   shortcut AUC are positively rank-correlated (Spearman ρ = +0.683, n = 9; +0.857 with the\n"
     "   original seven). High in-domain scores in these datasets therefore cannot be read as\n"
     "   evidence of morphological understanding.",
     "§1.3 贡献 2"),

    # ---- Discussion §2.1
    ("**First, high in-domain performance in standard pathology patch datasets is substantially\n"
     "inflated by non-morphological shortcuts.** In all seven sources, a logistic regression on image\n"
     "dimensions, file size, colour statistics, background and dark-pixel fractions — containing no\n"
     "tissue morphology whatsoever — predicts the label with AUC 0.72–0.98. The rank correlation\n"
     "between in-domain AUC and shortcut AUC is +0.857 (n = 7): the sources on which the encoder\n"
     "performs best are also those whose labels are most predictable from acquisition artefacts.",
     "**First, high in-domain performance in standard pathology patch datasets is substantially\n"
     "inflated by non-morphological shortcuts.** In all nine sources, a logistic regression on image\n"
     "dimensions, file size, colour statistics, background and dark-pixel fractions — containing no\n"
     "tissue morphology whatsoever — predicts the label with AUC 0.72–0.98. The rank correlation\n"
     "between in-domain AUC and shortcut AUC is +0.683 (n = 9), having been +0.857 with the original\n"
     "seven sources: the sources on which the encoder performs best tend also to be those whose\n"
     "labels are most predictable from acquisition artefacts, though the association is looser than\n"
     "the seven-source subset suggested.",
     "§2.1 第一条"),
    ("**Second, cross-source transfer collapses systematically, and the collapse is not specific to\n"
     "one encoder or to pathology-specific pretraining.** Across 42 source→target pairs and three\n"
     "frozen encoders, mean ΔAUC is −0.38 to −0.41, and 14–22 of the 42 pairs fall **below chance**\n"
     "(AUC < 0.5). Critically, an ImageNet-supervised encoder that has never seen histology collapses\n"
     "as badly or worse (22/42 below chance).",
     "**Second, cross-source transfer collapses systematically, and the collapse is not specific to\n"
     "one encoder or to pathology-specific pretraining.** Across 72 source→target pairs and three\n"
     "frozen encoders, mean ΔAUC is −0.36 to −0.38, and 22–31 of the 72 pairs fall **below chance**\n"
     "(AUC < 0.5). Critically, an ImageNet-supervised encoder that has never seen histology collapses\n"
     "as badly or worse (31/72 below chance).",
     "§2.1 第二条"),
    ("**Third, and most relevant clinically, domain-level AUC does not predict downstream usability.**",
     "**Third, and most relevant clinically, domain-level AUC does not predict downstream usability.**",
     "（占位，无改动）"),
]

# 度量值替换（正则，处理各处零散出现）
REGEX_EDITS: list[tuple[str, str, str]] = [
    (r"Spearman ρ = \+0\.857\)", "Spearman ρ = +0.683, n = 9)", "Abstract rho"),
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
        if old == new:
            continue
        if old in text:
            text = text.replace(old, new, 1)
            ok += 1
            print(f"  [OK]   {why}")
        else:
            miss += 1
            print(f"  [MISS] {why}")
    for pat, rep, why in REGEX_EDITS:
        new_text, n = re.subn(pat, rep, text)
        if n:
            text = new_text
            ok += 1
            print(f"  [OK]   {why}（{n} 处）")
        else:
            miss += 1
            print(f"  [MISS] {why}")

    stale = [m.group(0) for m in re.finditer(r"7 × 7|42 source|15/42|\+0\.857", text)]

    if args.dry_run:
        print(f"\n[dry] 命中 {ok}，未命中 {miss}；残留 {stale}")
        return 0

    DOC.write_text(text, encoding="utf-8")
    assert DOC.read_text(encoding="utf-8") == text
    print(f"\n已更新 {DOC.name}：命中 {ok}，未命中 {miss}；UTF-8 校验通过")
    print(f"残留待人工确认：{stale if stale else '无'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
