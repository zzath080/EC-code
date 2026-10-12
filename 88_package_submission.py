"""打包投稿用的交付包（zip）。

**为什么必须用 Python 而不是 PowerShell 的 Compress-Archive：**
    本会话已发生一次编码事故——用 PowerShell 的 `Add-Content`/`Set-Content` 写回
    `Methods_可直接引用.md`，87 处中文被写成替换字符且不可逆。
    Windows PowerShell 5.1 的 `Compress-Archive` 内部以 ANSI 处理条目名，
    对含中文路径/内容同样有风险。Python 的 `zipfile` 明确以 UTF-8 写条目名并原样
    复制字节，因此**打包环节必须绕开 PowerShell**。

包内结构：
    manuscript/     正文合并稿 + 分节稿 + 补充材料 + 投稿说明
    figures/        Figure 1-5（png + pdf）
    tables/         Table 1-8（csv + md）
    scripts/        生成与校验脚本（保证可复现）
    docs/           Data Availability + 图注

打包后立即校验：逐条读取 zip 内每个 .md/.csv，确认可被 UTF-8 解码。

用法：
    python scripts/88_package_submission.py
输出：
    dist/BSPC_submission_<日期>.zip
"""
from __future__ import annotations

import datetime as _dt
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"

MANU = ROOT / "docs" / "manuscript"
FIGS = ROOT / "docs" / "paper_figures"
TABLES = ROOT / "docs" / "paper_tables"
SCRIPTS = ROOT / "scripts"
DOCS = ROOT / "docs"

# (zip 内路径, 源路径)
MEMBERS: list[tuple[str, Path]] = [
    ("manuscript/Manuscript_Full.md", MANU / "Manuscript_Full.md"),
    ("manuscript/Supplementary_Material.md", MANU / "Supplementary_Material.md"),
    ("manuscript/Results_section.md", MANU / "Results_可直接引用.md"),
    ("manuscript/Methods_section.md", MANU / "Methods_可直接引用.md"),
    ("manuscript/Introduction_Discussion_sections.md",
     MANU / "Introduction_Discussion_可直接引用.md"),
    ("manuscript/Skeleton_navigation.md", MANU / "稿件骨架_Manuscript_Skeleton.md"),
    ("docs/Data_Availability.md", DOCS / "Data_Availability.md"),
    ("docs/Research_Design.md", DOCS / "主研究设计_跨来源稳健性基准.md"),
    ("docs/Figure_Table_Legends.md", FIGS / "FIGURE_TABLE_LEGENDS.md"),
    ("docs/Asset_Inventory.md", DOCS / "Asset_Inventory.md"),
    ("docs/Repo_Cleanup_Report.md", DOCS / "Repo_Cleanup_Report.md"),
]

FIG_STEMS = ["Figure1_cross_source_matrix", "Figure2_shortcut_audit",
             "Figure3_stain_normalization", "Figure4_external_biopsy",
             "Figure5_encoder_comparison"]
TABLE_STEMS = ["Table1_cohort_characteristics", "Table2_datasets",
               "Table3_cross_source_matrix", "Table4_shortcut_audit",
               "Table5_stain_ablation", "Table6_external_biopsy",
               "Table7_encoder_comparison", "Table8_encoder_by_experiment"]

KEY_SCRIPTS = [
    "source_registry.py", "10_tiling.py", "12_extract_features.py", "13_mil_train_eval.py",
    "21_make_paper_tables.py", "27_cross_specimen_transfer.py", "31_calibration_gap_test.py",
    "55_external_biopsy_validation.py", "59_stain_and_shortcut_audit.py",
    "60_cross_source_matrix.py", "63_calibration_methods.py", "64_make_figures_tables.py",
    "65_verify_results_doc.py", "73_encoder_comparison.py", "75_fix_broken_links.py",
    "76_stain_ablation_two_encoders.py", "77_extract_vitbasein_wsi.py",
    "80_regen_supplement_tables.py", "81_verify_supplement_numbers.py",
    "82_renumber_references.py", "86_relink_citations.py", "87_assemble_manuscript.py",
    "88_package_submission.py", "89_finalize_manuscript.py", "90_sync_skeleton_status.py",
    "91_shortcut_residualization.py", "92_probe_robustness.py",
    "93_trivial_baselines.py", "94_update_results_to9.py",
    "95_update_intro_disc_to9.py", "96_organize_repo.py",
    "97_package_assets.py", "98_verify_assets.py",
    "99_write_cleanup_report.py",
]

# 仓库根目录下的运行辅助文件（不在 scripts/ 内）
ROOT_FILES = [
    ("scripts/PYTHON_RUNTIME.md", ROOT / "PYTHON_RUNTIME.md"),
    ("scripts/README_project.md", ROOT / "README.md"),
]

# 包内说明：告诉接收方这是什么、怎么用、还差什么
MANIFEST_NOTE = """# 投稿交付包说明

**稿件题目**：Cross-source fragility of pathology foundation model embeddings:
a calibration- and shortcut-aware benchmark across organ and specimen types

**目标期刊**：Biomedical Signal Processing and Control (Elsevier, IF 4.9, JCR Q2)

---

## 1. 包内结构

```
manuscript/
  Manuscript_Full.md                  合并稿（Abstract -> References，可直接开始排版）
  Supplementary_Material.md           补充材料 Table S1-S8
  Results_section.md                  分节稿（写作时用，含每条结论的强制局限）
  Methods_section.md                  分节稿
  Introduction_Discussion_sections.md 分节稿 + 已核实文献表
  Skeleton_navigation.md              骨架与导航（含词数预警）
figures/    Figure1-5 .png (300 dpi) + .pdf (矢量)
tables/     Table1-8 .csv + .md
scripts/    生成与校验脚本
docs/       Data Availability、研究设计、图注
```

## 2. 运行环境（重要）

- 解释器必须用 **`D:\\Softwares\\Anaconda\\python.exe`**。本机 `python` 会解析到
  `C:\\Python314`，导致所有 cp312 wheel 报 ImportError。详见 `scripts/PYTHON_RUNTIME.md`。
- 依赖装在仓库 `.pylibs`，运行前需设 `PYTHONPATH`：
  ```
  $env:PYTHONPATH="<repo>\\.pylibs"; $env:PYTHONUTF8='1'
  & 'D:\\Softwares\\Anaconda\\python.exe' scripts\\65_verify_results_doc.py
  ```
- **不要用 PowerShell 的 `Set-Content`/`Add-Content` 编辑本包内任何 .md 文件**：
  它们默认按 ANSI/GBK 写入，会不可逆地损坏中文。本项目已因此损坏过一个文件。

## 3. 数字可复现性

所有正文引用的数字都能从 `runs/` 下的原始 JSON 重算：

| 脚本 | 作用 |
| --- | --- |
| `65_verify_results_doc.py` | 重算正文引用数字 + **过期数字护栏**（检出残留的上一代数值） |
| `81_verify_supplement_numbers.py` | 逐项核对补充材料 S2-S8 的每个数字（当前 210 项全通过） |
| `80_regen_supplement_tables.py` | 从 JSON 重新生成 S2-S8（**不要手改这些表**，密集小数手抄已两次出错） |
| `87_assemble_manuscript.py` | 从分节稿重新组装合并稿 |
| `86_relink_citations.py` | 按正文语义重排引用号并重建文献表 |

## 4. 投稿前仍需完成（4 项）

1. **词数**：合并稿英文约 7,360 词（含表格文本）。BSPC Full Paper 官方建议约 5,000 词，
   但**官方 Guide for Authors 在本项目环境中持续返回 HTTP 403，该数字未能核实**，
   可能只是软性指引。压缩方案（可省约 1,650 词）见分节稿的「词数说明」一节：
   把 Results 明细表移入补充材料、精简 Methods §2.9、合并 Discussion 局限条目。
2. **Abstract 已写**（227 词，满足 < 250）；**Highlights 已写 5 条**（每条 ≤ 85 字符，已实测）。
3. **代码/数据仓库链接**：`Data and Code Availability` 需填入 GitHub/Zenodo DOI。
4. **Methods §2.10 AI 使用声明**：BSPC 要求如实描述 AI 辅助工具的使用范围，需项目人确认口径。
5. **图注/表注**：`docs/Figure_Table_Legends.md` 为中英对照，投稿时取英文部分。

## 5. 引用完整性

- 文献表 21 条 + 数据引用 1 条（D1），**编号严格按正文出现顺序**。
- 每条文献的卷期页/DOI 均标注了核实方式：`Crossref 直查 DOI` 或 `Europe PMC 检索核实`，
  其余标 `既有综述 [VERIFIED]`（来自本项目早期文献综述，带核实状态标记）。
- 已做**逐条语义核对**：每个 `[n]` 的正文语境与其在文献表中的条目一致。
"""


def add(zf: zipfile.ZipFile, arc: str, src: Path, problems: list[str]) -> int:
    if not src.exists():
        problems.append(f"缺失：{src.relative_to(ROOT)}")
        return 0
    # 以二进制原样写入，避免任何编码转换
    zf.write(src, arcname=arc)
    return 1


def main() -> int:
    DIST.mkdir(exist_ok=True)
    stamp = _dt.date.today().isoformat()
    out = DIST / f"BSPC_submission_{stamp}.zip"
    problems: list[str] = []
    n = 0

    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # 包内说明用 writestr 直接写，避免额外落盘
        zf.writestr("README_DELIVERY.md", MANIFEST_NOTE)
        n += 1
        for arc, src in MEMBERS:
            n += add(zf, arc, src, problems)
        for arc, src in ROOT_FILES:
            n += add(zf, arc, src, problems)
        for stem in FIG_STEMS:
            for ext in (".png", ".pdf"):
                n += add(zf, f"figures/{stem}{ext}", FIGS / f"{stem}{ext}", problems)
        for stem in TABLE_STEMS:
            for ext in (".csv", ".md"):
                base = TABLES if stem.startswith(("Table1_", "Table2_")) else FIGS
                n += add(zf, f"tables/{stem}{ext}", base / f"{stem}{ext}", problems)
        for s in KEY_SCRIPTS:
            n += add(zf, f"scripts/{s}", SCRIPTS / s, problems)

    # ---- 打包后校验：zip 内每个文本文件都必须可被 UTF-8 解码
    bad: list[str] = []
    checked = 0
    with zipfile.ZipFile(out) as zf:
        for info in zf.infolist():
            if info.filename.lower().endswith((".md", ".csv", ".py", ".ps1")):
                checked += 1
                data = zf.read(info.filename)
                try:
                    data.decode("utf-8")
                except UnicodeDecodeError as e:
                    bad.append(f"{info.filename}: {e}")

    size_mb = out.stat().st_size / 1024 ** 2
    print(f"已打包 {out.relative_to(ROOT)}")
    print(f"  条目 {len(zipfile.ZipFile(out).infolist())} 个，写入 {n} 个，{size_mb:.1f} MB")
    print(f"  UTF-8 校验：检查 {checked} 个文本文件，失败 {len(bad)}")
    if bad:
        for b in bad[:20]:
            print("   -", b)
    if problems:
        print(f"  [WARN] 缺失 {len(problems)} 项：")
        for p in problems[:20]:
            print("   -", p)
    return 1 if (bad or problems) else 0


if __name__ == "__main__":
    sys.exit(main())
