#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""恢复 docs/manuscript/Methods_可直接引用.md。

背景：该文件曾被 PowerShell 的 `Set-Content`/`Add-Content` 按 ANSI/GBK 重写，
87 处中文字符损坏为替换字符，且 fs 层已拒绝以 UTF-8 读取。本脚本以
Python 的 `Path.write_text(..., encoding="utf-8")` 重写为合法的 UTF-8 文件
（Python 写文件不经过 PowerShell 的编码转换，因此不会二次损坏）。

用法：
    python scripts/79_restore_methods_doc.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "manuscript" / "Methods_可直接引用.md"

CONTENT = '''# Methods（可直接引用稿件段落）

> 本文件是 Methods 的成稿段落，与 `Results_可直接引用.md`、
> `Introduction_Discussion_可直接引用.md` 配套。
> **所有参数取自脚本实测值**（`scripts/10_tiling.py`、`12_extract_features.py`、
> `13_mil_train_eval.py`、`27_cross_specimen_transfer.py`、`31_calibration_gap_test.py`、
> `55_external_biopsy_validation.py`、`59/60/76`、`source_registry.py`），不是估计值。
> 引用格式按目标期刊要求预留为方括号数字 `[n]`，编号待全文合并时按出现顺序统一重排。
>
> **本文定位**：校准与评测方法学（calibration- and evaluation-methodology），
> 以临床可用性（固定阈值下的决策行为）为落点。写作时这条主线贯穿 Methods 的取舍。
>
> **⚠️ 编码事故记录**：本文件曾因用 PowerShell `Set-Content`/`Add-Content` 写回而损坏编码
> （87 处中文变为替换字符），已按要求重写。**不要用 PowerShell 的
> `Set-Content`/`Add-Content` 编辑本仓库的 UTF-8 中文文件**——它们默认按 ANSI/GBK 写入。
> 请改用文件编辑工具或 Python 的 `Path.write_text(..., encoding="utf-8")`，
> 并在写入后立即校验文件可被 UTF-8 解码。

---

## 2. Materials and methods

### 2.1 Study design and scope

We designed a **cross-source robustness benchmark** for pathology foundation-model
embeddings, evaluated at two levels of granularity:

1. **Patch level** — a fully crossed matrix over **seven** independent, openly licensed
   histopathology image datasets spanning six organ systems. Each dataset serves in turn as
   the training source and as the held-out target, giving 7 × 6 = **42 cross-source pairs**.
2. **Whole-slide (WSI) level** — an external validation on **91 real endometrial Pipelle
   biopsy** whole-slide images, plus a within-cohort cross-specimen-type comparison
   (frozen tissue sections to FFPE diagnostic slides) in TCGA-UCEC.

The design deliberately separates three questions that are usually conflated: (i) *is
in-domain performance trustworthy* (shortcut audit), (ii) *what happens across sources*
(cross-source matrix), and (iii) *does a high AUC imply clinical usability* (fixed-threshold
behaviour and calibration). We report no clinical efficacy endpoint: the biopsy cohort is
**entirely malignant**, so AUC, specificity and accuracy are **undefined** in the target
domain and are not reported (Section 2.8).

**Scope declaration.** We use **only openly available model weights**. Gated models
requiring institutional-email authorisation (UNI2-h, CONCH, Virchow2, H-optimus, Hibou)
were **not** evaluated, so all conclusions are scoped to openly accessible encoders. No
dataset requiring a proprietary vendor SDK was used.

### 2.2 Data sources

**Patch-level sources.** Seven datasets (Table 2), each restricted to those admitting an
explicit **binary** malignant versus non-malignant collapse. Datasets that cannot be reduced
to a binary label (segmentation masks, mitosis-detection annotations, organ-classification
sets) were **excluded** rather than coerced — this is why the final count is seven rather
than the eight initially targeted. Licences are recorded per source in
`docs/Data_Availability.md`.

Source definitions are centralised in a single registry (`scripts/source_registry.py`) shared
by the cross-source matrix and the shortcut audit, so that both analyses always operate on
the **same** source set. Labels are read from directory names, from filenames, or from a CSV,
depending on the source; the registry declares which.

> **⚠️ Mandatory limitation.** The *raw* class definitions differ between sources
> (e.g. normal epithelium versus adenocarcinoma; tumour versus non-tumour; non-cancerous
> versus four ovarian carcinoma subtypes; colon Cancer/Connective/Immune/Normal). Cross-source
> AUC is therefore a **composite** of domain shift *and* label-definition differences, and
> must not be read as a pure measure of domain shift. This is stated again in the Discussion
> and is not a fixable property of the present design.

**Whole-slide data.**

- **TCGA-UCEC** — 1,371 slide images (1,315 primary tumour / 54 solid-tissue normal /
  2 recurrent; 0.967 TB). Frozen tissue sections are `01A`/`01B`/`01C`, FFPE diagnostic
  slides are `01Z`, and solid-tissue-normal slides are `11A` (Table 1). The barcode-to-
  sample-type mapping was verified for all 1,371 slides by cross-tabulating the barcode
  against the GDC `sample_type` field.
- **ENDO-AID** — 91 Pipelle endometrial biopsy WSIs at 0.50 µm/px, **all malignant**,
  CC BY-NC 4.0. Used **only** for external validation of the malignant end.

### 2.3 Tiling

Tissue was detected on a thumbnail (long side ≤ 2,048 px) with a dual threshold
(grey < 220 **or** saturation > 20) followed by morphological opening, then tiles were
emitted on a 256 × 256 grid at **0.50 µm/px** (approximately 20×), matching the pretraining
resolution of the encoders. Only tiles with tissue fraction ≥ 0.5 were kept. A
low-magnification pre-screen followed by exact high-magnification re-checking was used.
**Only coordinate indices are stored** (`data/patches/`); patch images are not materialised
to disk.

Biopsy slides have a much lower tissue fraction than resection slides, so a dedicated
threshold set was used for tiling them (`scripts/54_tile_biopsy.py`); the default threshold
would have discarded every biopsy slide.

### 2.4 Encoders

Three **frozen** encoders were evaluated (Table 7). All accept 224 × 224 input with
ImageNet normalisation; features are the CLS token.

| Encoder | Architecture | Dim | Params | Pretraining |
| --- | --- | --- | --- | --- |
| `phikon-v2` | ViT-L/14 | 1,024 | 303.4 M | DINOv2 self-supervised, 160k+ WSIs |
| `phikon` | ViT-B/16 | 768 | 86.4 M | iBOT self-supervised, 40 M TCGA tiles |
| `vit-base-in` | ViT-B/16 | 768 | 86.4 M | ImageNet **supervised** (non-pathology control) |

`vit-base-in` is included as a **non-pathology control**: if the collapse appears there too,
it cannot be attributed to pathology-specific self-supervised pretraining. Encoder weights
were converted from `pytorch_model.bin` to `safetensors` where required, because current
`transformers` releases refuse `.bin` loading under `torch < 2.6` (CVE-2025-32434).

### 2.5 Patch-level protocol

**Probe.** A single L2-regularised logistic regression on frozen features (scikit-learn,
`max_iter = 2000`). We deliberately use one fixed, simple probe throughout so that
differences reflect the *embeddings*, not the classifier; **this matrix is therefore not a
model ranking**.

**In-domain evaluation.** 5-fold `GroupKFold`. Grouping is by **patient identifier where the
source provides one**, and by image filename stem otherwise. Only one of the seven sources
publishes a patient identifier (`breast_idc`; 279 patients), and it is the only source
evaluated under patient-level grouping. For the remaining six, same-patient leakage across
folds cannot be excluded — a limitation we state explicitly rather than leave implicit.

**Cross-source evaluation.** For each ordered pair (a, b), the probe is fitted on all of
source a and evaluated on all of source b. Reported: AUC, average precision, mean predicted
probability per class, and the gap ΔAUC = (in-domain AUC of b) minus (cross-source AUC of the
b-to-a direction), aggregated as specified in Results.

**Subsampling.** Two large sources are capped at 5,000 images per class for computational
feasibility (`breast_idc`, `colon_cancer_csv`); the other five are used at full size. The
cap is applied **per source per class**, preserving class balance, and per-source overrides
ensure the five original sources are never subsampled.

### 2.6 Shortcut audit

To test whether in-domain performance could be driven by non-morphological cues, we trained
a logistic regression on **image statistics only** — width, height, aspect ratio, file size,
per-channel RGB means and standard deviations, R/G and R/B ratios, near-white background
fraction, and dark-pixel fraction. No tissue morphology enters this model. Performance is
reported as 5-fold stratified cross-validated AUC with a cap of 500 images per class per
source (or the minority class size, whichever is smaller; realised values are 100–600 per
class, see Table 4). AUC near 0.5 indicates no usable shortcut; AUC substantially above 0.5
means the dataset's in-domain performance is **inflated** by acquisition artefacts.

This audit is **encoder-independent** by construction (it never touches an embedding), so it
is reported once rather than per encoder. We additionally report the **rank correlation**
between in-domain AUC and shortcut AUC across sources.

### 2.7 Stain-normalisation ablation

Reinhard colour normalisation [18] was applied and features were re-extracted. Two versions
are reported, and the distinction matters:

- **Historical version** (Results Section 3.5): normalised features were compared against the
  *full* original feature set, i.e. the two sides do not share an identical image set.
- **Controlled version** (Results Section 3.5b): a **single fixed stratified image subset**
  (at most 500 per source per class; 4,498 patches in total, drawn from the five original
  sources) was shared *identically* by all three encoders, and original and normalised
  features were extracted for each. ΔAUC is therefore **paired**.

The five-source restriction in the controlled version is deliberate: the subset predates the
addition of the last two sources, and holding it fixed keeps the ablation directly comparable
to the previously published values.

> **⚠️ Implementation limitation (must be stated).** Normalisation targets are computed
> **within each source** from that source's own mean and standard deviation. The procedure
> therefore removes *within-source* colour variation rather than aligning all sources to a
> common reference stain. We consequently claim only that **this implementation** does not
> reliably repair the collapse; we do not claim stain normalisation in general is
> ineffective. Macenko, Vahadane and adaptive colour deconvolution were not tested.

### 2.8 Whole-slide protocol

Slide-level classification used attention-based MIL (ABMIL) [19] on frozen features: a shared
encoder (`Linear(d to 256)`, ReLU, `Dropout(0.25)`), a gated attention network
(`Linear(d to 256)` with Tanh, gated by `Linear(d to 256)` with Sigmoid, then
`Linear(256 to 1)`), and a linear head. Optimiser **AdamW** (learning rate 1 × 10⁻³, weight
decay 1 × 10⁻⁴), cross-entropy loss, 30 epochs. At most **2,048** instances were sampled per
slide in both whole-slide experiments. Patient-level splits were used for training and were
verified to have **zero** patient overlap between train and test.

**Biopsy external validation.** Source domain = TCGA frozen tissue sections (malignant
`01A`/`01B`, benign `11A`; 97 slides with usable features). Target domain = 91 ENDO-AID
biopsies. The protocol was repeated over **three random seeds**.

**Cross-specimen-type comparison.** Within TCGA-UCEC only: frozen tissue sections (source) to
FFPE diagnostic slides `01Z` (target). This holds organ and resection setting fixed and
varies specimen processing.

> **⚠️ Structurally unidentifiable endpoint.** In TCGA-UCEC, malignant slides are FFPE
> diagnostic slides (`01Z`) and benign slides are frozen tissue slides (`11A`), so
> **specimen type and label are collinear** and a cross-specimen-type AUC is *not estimable*
> here. This is a data-design limitation, not a sample-size one. We therefore report only the
> **malignant-stratified** shift in predicted probability, which removes the class-imbalance
> confound.

### 2.9 Calibration and statistics

- **Discrimination** — AUC with bootstrap 95% CI (2,000 resamples); average precision.
- **Calibration** — expected calibration error (ECE, 10 equal-width bins), Brier score,
  reliability curves, and calibration-in-the-large (mean predicted probability minus observed
  positive rate).
- **Shift tests** — label-stratified permutation test (5,000 permutations; reported as
  (count + 1) / 5,001) on the difference in mean predicted probability between domains. The
  stratified form is used because the two domains differ in positive rate, which otherwise
  confounds ECE.
- **Calibration transfer** — temperature scaling and isotonic regression fitted on a 30%
  source-domain hold-out and applied to the target domain.
- **Multiplicity** — Holm correction for paired comparisons.
- **Reporting rule.** A performance difference is **not** claimed when the ΔAUC confidence
  interval crosses zero or when it does not survive Holm correction.

Two reporting traps we explicitly avoid, because both would be misleading:

1. **Ratios of ECE across encoders.** We do not quote an "N-fold increase in ECE", because the
   denominator is a different in-domain baseline for each encoder (0.0442 / 0.0994 / 0.1435,
   giving 5.83x / 2.72x / 3.55x for the cross-specimen-type shift). Only the **absolute**
   target-domain ECE (0.2577 / 0.2699 / 0.5099) is comparable.
2. **Spread comparisons across different baselines.** An earlier draft described biopsy
   prediction spread as increasing approximately 19-fold; that used a *mixed-class*
   source-domain test set as baseline (SD 0.0062). Against the like-for-like baseline (spread
   of source-domain **malignant** predictions, SD 0.1132) the biopsy spread (0.1156) is
   essentially unchanged (1.02x). We therefore report the **mean-probability drop** as the
   calibration evidence and do not use spread as evidence of instability.

### 2.10 AI-assisted tooling

*[To be completed before submission — the target journal requires that any use of AI or
AI-assisted tools in the research design or methods be described reproducibly. State exactly
which tool was used for what: literature triage, code drafting, documentation, or analysis
support, and confirm that no AI tool generated or altered any figure.]*

---

## 附：Methods 与脚本的对应关系（供复核，不进正文）

| Methods 小节 | 脚本 | 主要产物 |
| --- | --- | --- |
| 2.2 数据清单与来源定义 | `source_registry.py`、`02_data_inventory.py`、`21_make_paper_tables.py` | Table 1、Table 2、`slide_master.csv` |
| 2.3 分块 | `10_tiling.py`、`54_tile_biopsy.py` | `data/patches/`、`data/qc/tiling_qc_*.csv` |
| 2.4 编码器 | `11_fetch_models.py`、`70_fetch_second_encoder.py`、`72_bin_to_safetensors.py` | `models/` |
| 2.5 patch 协议 | `60_cross_source_matrix.py` | Figure 1、Table 3、Table 7 |
| 2.6 捷径审计 | `59_stain_and_shortcut_audit.py --stats` | Figure 2、Table 4 |
| 2.7 染色消融 | `59 ... --normalize`、`76_stain_ablation_two_encoders.py` | Figure 3、Table 5、Table 8 |
| 2.8 WSI 协议 | `27_cross_specimen_transfer.py`、`55_external_biopsy_validation.py` | Figure 4、Table 6 |
| 2.9 统计 | `31_calibration_gap_test.py`、`63_calibration_methods.py` | 校准迁移结果 |
| 数字核对 | `65_verify_results_doc.py` | 从原始 JSON 重算所有引用数字 + 过期数字护栏 |
'''


def main() -> int:
    if DOC.exists():
        try:
            DOC.read_text(encoding="utf-8")
            print("[INFO] 目标文件当前是合法 UTF-8；仍将按最新内容重写。")
        except UnicodeDecodeError:
            print("[INFO] 目标文件当前不是合法 UTF-8（曾损坏），将重写修复。")
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text(CONTENT, encoding="utf-8")

    # 写后立即校验：这是本次事故的直接教训——写完必须验证可解码
    back = DOC.read_text(encoding="utf-8")
    ok = back == CONTENT
    print(f"已写入 {DOC.relative_to(ROOT)}")
    print(f"UTF-8 往返校验: {'通过' if ok else '不一致'}")
    print(f"字符数 {len(back)}，行数 {len(back.splitlines())}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
