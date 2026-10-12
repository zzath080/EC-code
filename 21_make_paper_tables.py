#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
21_make_paper_tables.py
=======================
从已核实的 manifest / 临床 / 数据集统计生成论文表格（Markdown + CSV）。

产出：
    docs/paper_tables/Table1_cohort_characteristics.md
    docs/paper_tables/Table2_datasets.md
    docs/paper_tables/Table1_cohort_characteristics.csv
    docs/paper_tables/Table2_datasets.csv
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANI = ROOT / "data" / "manifests"
OUT = ROOT / "docs" / "paper_tables"


def load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def table1() -> tuple[list[list[str]], list[list[str]]]:
    clinical = load_json(MANI / "tcga_ucec_clinical_summary.json")
    slides = load_json(MANI / "tcga_ucec_summary.json")

    rows: list[list[str]] = []
    csv_rows: list[list[str]] = []

    def add(field: str, value: str, source: str) -> None:
        rows.append([field, value, source])
        csv_rows.append([field, value, source])

    add("**队列规模**", "", "")
    add("病例数（有病理诊断记录）", str(clinical.get("n_cases", "—")), "GDC cases API")
    if slides:
        by = slides.get("by_sample_type", {})
        total = slides.get("total_slides", 0)
        add("切片总数（Slide Image）", str(total), "GDC files API")
        add("　原发瘤切片", str(by.get("Primary Tumor", {}).get("slides", "—")), "GDC files API")
        add("　实性正常组织切片", str(by.get("Solid Tissue Normal", {}).get("slides", "—")), "GDC files API")
        add("　复发瘤切片", str(by.get("Recurrent Tumor", {}).get("slides", "—")), "GDC files API")
        add("切片总体量", f"{slides.get('total_size_tb', 0)} TB", "GDC files API")
        codes = slides.get("sample_type_code_counts", {})
        # 条码 → 组织类型为**实测**（slide_master.csv 交叉核验）：
        #   01A/01B/01C = Primary Tumor（735/11/4，共 750）；01Z = Primary Tumor（565）；
        #   11A = Solid Tissue Normal（54）；02A/02Z = Recurrent Tumor（各 1）。
        # 此前只列 01Z/01A/11A，三者之和 1354 ≠ 1371，表内"消失 17 张"；
        # 且把 01A 标成"组织切片"会让读者以为 750 张冷冻切片只有 735 张。
        add("　01Z（FFPE 诊断切片）", str(codes.get("01Z", 0)), "文件名条码解析")
        add("　01A（冰冻/TSA 组织切片）", str(codes.get("01A", 0)), "文件名条码解析")
        add("　01B / 01C（组织切片，其他份号）",
            f"{codes.get('01B', 0)} / {codes.get('01C', 0)}", "文件名条码解析")
        add("　11A（实性正常组织）", str(codes.get("11A", 0)), "文件名条码解析")
        add("　02A / 02Z（复发瘤）",
            f"{codes.get('02A', 0)} / {codes.get('02Z', 0)}", "文件名条码解析")
        _sum_known = sum(codes.values())
        add("　条码合计（应等于切片总数）", str(_sum_known), "文件名条码解析")
        add("同时具备肿瘤与正常组织的病例", str(slides.get("patients_with_normal_and_tumor", "—")), "GDC files API")

    add("**人口学**", "", "")
    age = clinical.get("age_stats", {})
    if age.get("n"):
        sd = age.get("sd")
        sd_txt = f"{sd}" if sd is not None else "—"
        add("年龄（岁，n=%d）" % age["n"],
            f"{age.get('mean')} ± {sd_txt}（范围 {age.get('min')}–{age.get('max')}）",
            "GDC demographic.age_at_index")
    vs = clinical.get("vital_status", {})
    if vs:
        add("生存状态", "；".join(f"{k} {v}" for k, v in vs.items()), "GDC demographic.vital_status")

    add("**组织学**", "", "")
    grade = clinical.get("tumor_grade", {})
    if grade:
        add("组织学分级", "；".join(f"{k} {v}" for k, v in grade.items()), "GDC diagnoses.tumor_grade")
    dx = clinical.get("primary_diagnosis", {})
    if dx:
        top = list(dx.items())[:5]
        # GDC 里真实存在"字段为空字符串"的记录（12 例）。直接打印会变成一个不可见的空格，
        # 读者会以为是排版错误——必须显式标注它是缺失值。
        def _lbl(k: str) -> str:
            return "未记录（GDC 空字段）" if k.strip() == "" else k
        add("主要诊断（前 5）",
            "；".join(f"{_lbl(k)} {v}" for k, v in top), "GDC diagnoses.primary_diagnosis")

    add("**分期**", "", "")
    figo = clinical.get("figo_stage", {})
    if figo:
        add("FIGO 分期", "；".join(f"{k} {v}" for k, v in list(figo.items())[:10]), "GDC diagnoses.figo_stage")

    return rows, csv_rows


def table2() -> tuple[list[list[str]], list[list[str]]]:
    """数据集对照表（数字均为实测；**不含任何需专有 SDK 的数据集**）。"""
    header = ["数据集", "标本类型", "切片/图像数", "标签体系", "格式", "体量", "获取方式", "本文角色"]
    data = [
        ["TCGA-UCEC", "手术切除", "1371 张切片（肿瘤 1315 / 正常 54 / 复发 2）",
         "癌 vs 实性正常组织；无良性病变谱系", "SVS", "0.967 TB", "GDC 公开，无需授权", "源域（切除）"],
        ["ENDO-AID (Radboud)", "Pipelle 活检", "91 张切片", "**全部为癌**，无良性标签",
         "TIFF @0.5 µm/px", "16.9 GB（解压后 49.5 GB）", "Zenodo 7372187，CC BY-NC 4.0",
         "**外部活检域验证（仅恶性端）**"],
        ["Endometrial Histopathology (HIENet)", "patch", "3302 张",
         "正常 NE 1333 / 息肉 EP 636 / 增生 EH 798 / 腺癌 EA 535", "JPEG", "197 MB",
         "Kaggle，CC0", "跨来源来源之一（子宫内膜）"],
        ["Uterus Cancer (HISTOPANTUME 子集)", "patch", "6334 张",
         "non-tumour 3334 / tumour 3000", "JPEG", "130 MB", "Kaggle，CC0", "跨来源来源之一（子宫）"],
        ["Lung & Colon Cancer", "patch", "25000 张",
         "colon_n 5000 / lung_n 5000 / colon_aca 5000 / lung_aca 5000 / lung_scc 5000",
         "JPEG", "1.80 GB", "Kaggle，CC BY-SA 4.0", "跨来源来源之一（肺+结肠）"],
        ["BreakHis 400×", "patch", "1693 张", "benign 547 / malignant 1146", "PNG", "803 MB",
         "Kaggle，CC0", "跨来源来源之一（乳腺）"],
        ["Ovarian Cancer & Subtypes", "patch", "498 张",
         "非癌 100 / 透明细胞 100 / 内膜样 98 / 黏液性 100 / 浆液性 100", "JPEG/PNG", "83 MB",
         "Kaggle，CC BY-SA 4.0", "跨来源来源之一（卵巢）"],
        # ---- 扩展来源（用于把矩阵从 5 个来源扩到 7 个） ----
        ["Breast Histopathology Images (IDC)", "patch", "555 048 张（每类抽 5000 用于评估）",
         "IDC 阳性（class1，恶性）157 572 / IDC 阴性（class0，良性）397 476",
         "PNG 50×50 px", "3.17 GB", "Kaggle，CC0",
         "跨来源来源之一（乳腺，**唯一带患者 ID**：279 名患者，可按患者分组）"],
        ["Colon Cancer (CSV 标签)", "patch", "2190 张（每类抽 546 用于评估）",
         "Cancer 546 / Connective 726 / Immune 729 / Normal 189（后三类折叠为良性）",
         "PNG", "224 MB", "Kaggle，Apache-2.0",
         "跨来源来源之一（结肠；恶性仅占 7.2%）"],
    ]
    rows = [header] + data
    return data, rows


def md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "| " + " | ".join("---" for _ in header) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    r1, c1 = table1()
    md1 = [
        "# Table 1. Characteristics of the TCGA-UCEC cohort",
        "",
        "> 数据来源：GDC REST API 实测（`scripts/09_build_clinical_manifest.py`、`scripts/00_build_gdc_manifest.py`）。",
        "> 缺失值以 GDC 原始记录为准，未做插补。",
        "",
        "| 特征 | 值 | 数据来源 |",
        "| --- | --- | --- |",
    ]
    for a, b, c in r1:
        if b:
            md1.append(f"| {a} | {b} | {c} |")
        else:
            # 小节标题行：去掉加粗标记，避免生成 **** 这样的嵌套标记
            md1.append(f"| **{a.strip('*')}** | | |")
    (OUT / "Table1_cohort_characteristics.md").write_text("\n".join(md1), encoding="utf-8")
    with open(OUT / "Table1_cohort_characteristics.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["feature", "value", "source"])
        w.writerows(c1)

    d2, r2 = table2()
    md2 = [
        "# Table 2. Public histopathology datasets used in this study",
        "",
        "> 所有数字为实测（来自各获取脚本的运行输出），非估计值。",
        "> 本表**不含**任何需专有 SDK 才能解码的数据集。",
        "> 说明：命名 \"ENDO-AID\" 指 Radboud 的 91 张全癌挑战赛数据集（本表中作为外部活检域验证）。",
        "",
        "| 数据集 | 标本类型 | 切片/图像数 | 标签体系 | 格式 | 体量 | 获取方式 | 本文角色 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in d2:
        md2.append("| " + " | ".join(str(c).replace("\n", "<br>") for c in r) + " |")
    (OUT / "Table2_datasets.md").write_text("\n".join(md2), encoding="utf-8")
    with open(OUT / "Table2_datasets.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "specimen_type", "n_slides", "labels", "format", "size", "access", "role"])
        w.writerows(d2)

    print(f"Table 1: {len(c1)} 行 -> {OUT/'Table1_cohort_characteristics.md'}")
    print(f"Table 2: {len(d2)} 个数据集 -> {OUT/'Table2_datasets.md'}")
    print()
    print("\n".join(md1[:14]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
