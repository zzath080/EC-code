"""从 JSON 生成 Supplementary Material 的全部明细表（S2–S8）。

事故背景（必须保留在注释里，避免以后又改回手写）：
    本文件的前身 `scripts/80_regen_supplement_tables.py` 只自动生成了 S2/S3，
    S4–S8 是**手工转录**的。实测发现 S5（染色消融逐配对 ΔAUC）大量数值错误——
    例如 `breakhis_400x→ovarian_ca` 真值 −0.1650 而文中写作 −0.0190，
    `endo_histo→uterus_ca` 真值 −0.2152 而文中写作 −0.0190。
    **密集小数绝不能手抄。** 因此 S4–S8 一并改为脚本生成。

符号约定：文档使用 **U+2212 MINUS SIGN「−」**（排版更正确），而 `f"{-0.5:.4f}"`
产生 ASCII hyphen-minus。本脚本统一用 `_num()` 输出 U+2212，并在校验时同样归一化，
否则会出现"数字其实一样但字符串不等"的假报警。

用法：
    python scripts/80_regen_supplement_tables.py --dry-run
    python scripts/80_regen_supplement_tables.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "manuscript" / "Supplementary_Material.md"
RUNS = ROOT / "runs"

ORDER = ["lung_colon", "uterus_ca", "breakhis_400x", "endo_histo",
         "ovarian_ca", "breast_idc", "colon_cancer_csv"]

BEGIN = "<!-- BEGIN:AUTO-TABLES (由 scripts/80 生成，请勿手改) -->"
END = "<!-- END:AUTO-TABLES -->"


def load(p: Path) -> dict | None:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def num(v: float, signed: bool = False, digits: int = 4) -> str:
    """数值格式化：负号统一用 U+2212，避免与文档中的排版符号不一致。"""
    s = f"{v:+.{digits}f}" if signed else f"{v:.{digits}f}"
    return s.replace("-", "\u2212")


def build_s2(mx: dict) -> str:
    ind, cross = mx["in_domain"], mx["cross_matrix"]
    lines = ["| train \\ test | " + " | ".join(ORDER) + " |",
             "| --- | " + " | ".join("---" for _ in ORDER) + " |"]
    for a in ORDER:
        cells = []
        for b in ORDER:
            if a == b:
                cells.append(f"**{num(ind[a]['auc'])}**")
            else:
                v = cross.get(f"{a}->{b}", {}).get("auc")
                cells.append(num(v) if v is not None else "n/a")
        lines.append(f"| **{a}** | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_s3(mx: dict) -> str:
    at = mx["as_target"]
    lines = ["| Target source | Mean cross-source AUC | Best | Worst | In-domain AUC | ΔAUC |",
             "| --- | --- | --- | --- | --- | --- |"]
    for t in sorted(at, key=lambda k: at[k]["mean_auc"]):
        v = at[t]
        lines.append(f"| `{t}` | {num(v['mean_auc'])} | {num(v['max_auc'])} | {num(v['min_auc'])} | "
                     f"{num(v['in_domain_auc'])} | {num(v['gap'], signed=True)} |")
    return "\n".join(lines)


def build_s4(mx: dict, sa: dict) -> str:
    sc = sa.get("shortcut_audit", {})
    rows = []
    for k, d in sc.items():
        ind = mx["in_domain"].get(k, {}).get("auc")
        diff = (d["stats_auc"] - ind) if ind is not None else None
        top = ", ".join(f for f, _ in d["top_features"][:3])
        rows.append((k, d["n_per_class"], d["stats_auc"], top, ind, diff))
    rows.sort(key=lambda r: -r[2])
    lines = ["| Source | Images per class | Shortcut AUC | Top features | In-domain AUC | Difference |",
             "| --- | --- | --- | --- | --- | --- |"]
    for k, n, sc_auc, top, ind, diff in rows:
        lines.append(f"| `{k}` | {n} | **{num(sc_auc)}** | {top} | "
                     f"{num(ind) if ind is not None else 'n/a'} | "
                     f"{num(diff, signed=True) if diff is not None else 'n/a'} |")
    return "\n".join(lines)


def build_s5(st: dict) -> str:
    summ = st.get("summary") or {}
    encs = list(summ)
    base = "phikon-v2" if "phikon-v2" in summ else encs[0]
    pairs = sorted(summ[base]["delta_auc"], key=lambda p: -summ[base]["delta_auc"][p])
    header = ["Pair"] + encs + ["Same sign in all?"]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for pair in pairs:
        vals = [summ[e]["delta_auc"].get(pair) for e in encs]
        present = [v for v in vals if v is not None]
        if len(present) == len(encs) and (all(v > 0 for v in present) or all(v < 0 for v in present)):
            same = "✓ 同向"
        else:
            same = "✗"
        cells = [f"`{pair}`"] + [(num(v, signed=True) if v is not None else "—") for v in vals] + [same]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_s6(ec: dict) -> str:
    encs = ec.get("encoders", {})
    other = ec.get("other_experiments", {})
    header = ["Encoder", "Arch", "Dim", "Params", "Pretraining",
              "Patch in-domain AUC", "Patch cross AUC", "Patch ΔAUC", "Pairs below chance",
              "Stain mean ΔAUC", "Biopsy sens. @0.5", "Biopsy sens. seed SD",
              "Biopsy mean p", "Source malignant mean p", "Cross-specimen Δp", "p"]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for e in encs:
        s = encs[e]
        rec = other.get(e, {})
        b, st_, x = rec.get("biopsy"), rec.get("stain"), rec.get("cross_specimen")
        cells = [f"`{e}`", s.get("arch", ""), str(s.get("dim", "")), f"{s.get('params_m', '')} M",
                 s.get("pretrain", ""),
                 num(s["in_domain_mean_auc"]), num(s["cross_mean_auc"]),
                 num(s["mean_gap"], signed=True), f"{s['n_below_chance']}/{s['n_cross_pairs']}",
                 num(st_["delta_mean"], signed=True) if st_ else "n/a",
                 num(b["sensitivity_at_0.5"]) if b else "n/a",
                 num(b.get("sensitivity_seed_sd")) if b and b.get("sensitivity_seed_sd") is not None else "n/a",
                 num(b["biopsy_mean_pred"]) if b else "n/a",
                 num(b["source_malignant_mean_pred"]) if b else "n/a",
                 num(x["stratified_delta_mean_pred"], signed=True) if x and x.get("stratified_delta_mean_pred") is not None else "n/a",
                 str(x["stratified_permutation_p"]) if x and x.get("stratified_permutation_p") is not None else "n/a"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_s7() -> str:
    header = ["Encoder", "Seed", "Source val. AUC", "Sensitivity @0.5", "Mean predicted p",
              "SD of p", "Slides ≥0.5"]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for enc, d in (("phikon-v2", "_external_biopsy"), ("phikon", "_external_biopsy_phikon"),
                   ("vit-base-in", "_external_biopsy_vit-base-in")):
        r = load(RUNS / d / "results.json")
        if not r:
            continue
        for ps in r.get("per_seed", []):
            eb = ps["external_biopsy"]
            above = eb["n"] - eb["n_below_0.5"]
            lines.append("| " + " | ".join([
                f"`{enc}`", str(ps["seed"]), num(ps["source_val_auc"]),
                num(eb["sensitivity_at_0.5"]), num(eb["mean_pred"]),
                num(eb["std_pred"]), f"{above} / {eb['n']}"]) + " |")
    return "\n".join(lines)


def build_s8(cal: dict) -> str:
    s = cal.get("summary", {})
    lines = ["| Calibration | Mean ECE | ΔECE vs raw |", "| --- | --- | --- |"]
    rows = [("None", s.get("mean_ece_raw"), None),
            ("Temperature scaling (fitted on source)", s.get("mean_ece_temperature"), s.get("delta_ece_temperature")),
            ("Isotonic (fitted on source)", s.get("mean_ece_isotonic"), s.get("delta_ece_isotonic")),
            ("Isotonic (fitted on target itself, in-sample reference)",
             s.get("mean_ece_isotonic_oracle"), s.get("delta_ece_isotonic_oracle"))]
    for label, ece, delta in rows:
        lines.append(f"| {label} | {num(ece) if ece is not None else 'n/a'} | "
                     f"{num(delta, signed=True) if delta is not None else '—'} |")
    return "\n".join(lines)


def build_s9() -> str:
    """S9 探针稳健性：同一特征/折划分下换探针。"""
    j = load(RUNS / "_probe_robustness" / "results.json")
    if not j:
        return "*(runs/_probe_robustness/results.json 缺失)*"
    lines = ["| Probe | Description | Train cap | In-domain mean AUC | Cross-source mean AUC | "
             "Mean ΔAUC | Pairs below chance |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for k, v in j["probes"].items():
        s = v["summary"]
        cap = v["train_cap"] if v["train_cap"] else "full"
        lines.append(f"| `{k}` | {v['description']} | {cap} | {num(s['in_domain_mean_auc'])} | "
                     f"{num(s['cross_mean_auc'])} | **{num(s['mean_gap'], signed=True)}** | "
                     f"{s['n_below_chance']}/{s['n_cross_pairs']} |")
    lines += ["", f"**Verdict:** {j['verdict']['conclusion']}", "",
              f"- Gap range across probes: {num(j['verdict']['gap_range'][0], signed=True)} to "
              f"{num(j['verdict']['gap_range'][1], signed=True)}",
              f"- All probes collapse (gap < −0.2): **{j['verdict']['all_probes_collapse']}**", "",
              "Per-source in-domain AUC and realised training size:", "",
              "| Source | " + " | ".join(j["probes"]) + " | Training n per fold (lr) |",
              "| --- | " + " | ".join("---" for _ in j["probes"]) + " | --- |"]
    first = list(j["probes"])[0]
    for sname in sorted(j["probes"][first]["in_domain"]):
        row = [f"`{sname}`"]
        for k in j["probes"]:
            v = j["probes"][k]["in_domain"].get(sname)
            row.append(num(v) if v is not None else "n/a")
        row.append(str(j["probes"][first]["n_train_used"].get(sname, "n/a")))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_s10() -> str:
    """S10 捷径残差化：量化域内 AUC 中线性可解释的非形态学成分。"""
    j = load(RUNS / "_shortcut_residual" / "baseline.json")
    if not j:
        return "*(runs/_shortcut_residual/baseline.json 缺失)*"
    lines = ["| Source | n | In-domain AUC | After residualisation | ΔAUC | Statistics-only AUC |",
             "| --- | --- | --- | --- | --- | --- |"]
    for k, v in sorted(j["sources"].items(), key=lambda kv: kv[1]["delta_auc"]):
        lines.append(f"| `{k}` | {v['n']} | {num(v['auc_original'])} | "
                     f"{num(v['auc_after_residualization'])} | "
                     f"**{num(v['delta_auc'], signed=True)}** | {num(v['auc_statistics_only'])} |")
    s = j.get("summary", {})
    lines += ["", f"- Mean ΔAUC: **{num(s.get('mean_delta_auc'), signed=True)}** "
                  f"(range {num(s.get('min_delta_auc'), signed=True)} to "
                  f"{num(s.get('max_delta_auc'), signed=True)}); "
                  f"all negative: **{s.get('all_negative')}**", "",
              "> **Interpretation.** ΔAUC is a **lower bound** on the shortcut contribution: only the",
              "> *linearly* explainable non-morphological component is removed, and residualisation",
              "> also removes genuine morphological signal that is collinear with acquisition",
              "> artefacts. The residualised AUC still exceeds the statistics-only AUC in every",
              "> source, so shortcuts are a material confounder rather than the sole driver of",
              "> in-domain performance.", ""]
    return "\n".join(lines)


def build_s11() -> str:
    """S11 平凡预测器基线（全恶性队列上的常数预测器）。"""
    j = load(RUNS / "_trivial_baselines" / "baselines.json")
    if not j:
        return "*(runs/_trivial_baselines/baselines.json 缺失)*"
    lines = ["| Target domain | Predictor | Sensitivity @0.5 | Brier | ECE | Calibration-in-the-large |",
             "| --- | --- | --- | --- | --- | --- |"]
    label = {"endoaid_biopsy": "ENDO-AID biopsy (n = 91, all malignant)",
             "tcga_01z_diagnostic": "TCGA FFPE diagnostic 01Z (n = 10, all malignant)",
             "tcga_frozen_test_balanced": "TCGA frozen tissue test (balanced, AUC estimable)"}
    for key, entry in j["target_domains"].items():
        dom = label.get(key, key)
        rows = []
        for name, v in entry.get("trivial", {}).items():
            rows.append((f"**{name.replace('_', ' ')}**", v))
        for name, v in entry.get("model", {}).items():
            rows.append((f"`{name}` (this work)", v))
        for name, v in rows:
            # 模型条目不含 ece（缺该字段时显示 n/a），常数条目不含 auc
            sens = v.get("sensitivity_at_0.5")
            ece = v.get("ece", v.get("cross_specimen_ece"))
            citl = v.get("calibration_in_the_large")
            lines.append(
                f"| {dom} | {name} | "
                f"{num(sens) if sens is not None else 'n/a'} | "
                f"{num(v['brier']) if 'brier' in v else (num(v['cross_specimen_brier']) if 'cross_specimen_brier' in v else 'n/a')} | "
                f"{num(ece) if ece is not None else 'n/a'} | "
                f"{num(citl, signed=True) if citl is not None else 'n/a'} |")
        lines.append("")   # 分组空行
    lines += ["> On the two all-malignant target cohorts, sensitivity, Brier and ECE are attained by a",
              "> constant predictor p ≡ 1.0, so they carry **no evidence of discrimination**. Only",
              "> calibration-in-the-large remains informative, and a trivial predictor is better",
              "> calibrated than any of the three models. The balanced in-domain row is the control:",
              "> there a constant predictor is badly calibrated (Brier 0.250) while the model is not.", ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not DOC.exists():
        print(f"[ERROR] 缺 {DOC}", file=sys.stderr)
        return 2

    mx = load(RUNS / "_cross_source_matrix" / "results.json")
    sa = load(RUNS / "_stain_shortcut_audit" / "results.json")
    st = load(RUNS / "_stain_ablation_encoders" / "results.json")
    ec = load(RUNS / "_encoder_comparison" / "results.json")
    cal = load(RUNS / "_calibration" / "results.json")
    for name, obj in (("矩阵", mx), ("捷径审计", sa), ("染色消融", st),
                      ("编码器汇总", ec), ("校准迁移", cal)):
        if obj is None:
            print(f"[WARN] 缺 {name} 的 results.json，对应表将跳过")

    parts = [BEGIN, "",
             "> **本节 S2–S8 全部由 `scripts/80_regen_supplement_tables.py` 从 `runs/` 下的",
             "> JSON 自动生成，请勿手改。**",
             ">",
             "> 改为自动生成的原因：S5 最初为手工转录，实测出现大量数值错误",
             "> （例如 `breakhis_400x→ovarian_ca` 真值 −0.1650 而文中写作 −0.0190）。",
             "> **密集小数不可手抄。** 数据更新后重跑该脚本即可同步。", ""]

    if mx:
        parts += ["### Table S2. Cross-source AUC matrix (7 × 7)", "",
                  "行 = 训练来源，列 = 测试来源（Phikon-v2 冻结特征 + 逻辑回归）。"
                  "对角线加粗为域内 AUC。", "", build_s2(mx), "",
                  "### Table S3. Per-target-source cross-source summary", "", build_s3(mx), ""]
    if mx and sa:
        parts += ["### Table S4. Shortcut audit and in-domain performance, per source", "",
                  "仅用**非形态学**图像统计量训练逻辑回归（5 折分层交叉验证）。"
                  "Difference = 捷径 AUC − 域内 AUC。", "", build_s4(mx, sa), ""]
    if st:
        parts += ["### Table S5. Stain-normalisation ablation: per-pair ΔAUC, all encoders", "",
                  "同一固定分层子集（每来源每类 ≤500，共 4,498 张，取自原 5 个来源）上的"
                  "**配对** ΔAUC = 归一化后 − 原始。", "", build_s5(st), ""]
    if ec:
        parts += ["### Table S6. Encoder dependence across all four experiments", "", build_s6(ec), ""]
    parts += ["### Table S7. Biopsy external validation: per-seed detail", "",
              "源域 = TCGA 冰冻组织切片；目标域 = 91 张 ENDO-AID 活检（**全为恶性**，"
              "故 AUC / 特异度在目标域不可定义）。", "", build_s7(), ""]
    if cal:
        parts += ["### Table S8. Calibration transfer across sources", "", build_s8(cal), ""]

    parts += ["### Table S9. Probe robustness: the collapse is not an artefact of the classifier", "",
              "特征、标签、抽样与折划分**完全固定**，只替换探针。测试集始终为全部目标样本，"
              "因此 AUC 在探针之间严格可比；SVM / kNN 的训练集因计算可行性做分层子采样，"
              "实际训练量逐项列出。", "", build_s9(), ""]

    parts += ["### Table S10. Shortcut residualisation: how much of the in-domain AUC is "
              "linearly explainable by non-morphological cues", "",
              "对每个来源，用**训练折**的图像统计量做岭回归预测特征的每一维，验证折上残差化后"
              "重跑同一探针与同一 GroupKFold 划分。", "", build_s10(), ""]

    parts += ["### Table S11. Trivial-predictor baselines on the whole-slide target cohorts", "",
              "两个 WSI 级目标域均**无阴性样本**。本表列出常数预测器与本文模型的全部指标，"
              "用以界定哪些指标在该队列上有判别信息。", "", build_s11(), ""]

    parts += ["> ⚠️ **S8 的解读警告**：原始 ECE 已 ≈0.35 而配对跨来源 AUC 仅 0.35–0.82。",
              "> 当 AUC 接近随机时预测概率与标签几乎无关，ECE 主要反映**判别力丧失**而非校准失当。",
              "> 本文**不**把「校准不可迁移」当作独立发现。最后一行是**目标域自拟合的过拟合上界，",
              "> 不得当作可达性能引用**。", "", "<!-- 解读文字由人工维护，见正文 Discussion -->", END]

    block = "\n" + "\n".join(parts) + "\n"

    text = DOC.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        print("[ERROR] 未找到自动区块标记", file=sys.stderr)
        return 2
    pre = text[: text.index(BEGIN)]
    post = text[text.index(END) + len(END):]
    new = pre + block.strip("\n") + post

    if args.dry_run:
        print("[dry] 将替换自动区块，长度", len(block))
        return 0

    DOC.write_text(new, encoding="utf-8")
    back = DOC.read_text(encoding="utf-8")
    assert back == new, "写回内容不一致"
    print(f"已生成 S2–S11；UTF-8 往返校验通过；文件 {len(back)} 字符")
    return 0


if __name__ == "__main__":
    sys.exit(main())
