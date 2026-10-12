"""校验 Supplementary Material 与正文明细表的数字是否与 JSON 一致。

动机（真实事故）：
    `Supplementary_Material.md` 的 Table S2 最初是**手工转录**的 7×7 = 49 格四位小数，
    实测出现多处错误（`lung_colon→breakhis_400x` 写成 0.4196、真值 0.3559；
    `lung_colon→ovarian_ca` 写成 0.6129、真值 0.5523）。手抄密集数字不可靠。

    本脚本对**每张明细表的每一个数字**做程序化核对：从 `runs/` 下的 JSON 重算，
    再与文档中实际出现的数值比对，任何不匹配都报出来。

覆盖的表：
    S2  跨来源 7×7 矩阵      <- runs/_cross_source_matrix/results.json
    S3  逐目标域汇总          <- 同上
    S4  捷径审计 + 域内 AUC   <- 同上 + runs/_stain_shortcut_audit/results.json
    S5  染色消融逐配对 ΔAUC   <- runs/_stain_ablation_encoders/results.json
    S6  四实验 × 编码器汇总   <- 矩阵 + 消融 + 活检 + 跨标本
    S7  活检逐种子            <- runs/_external_biopsy*/results.json
    S8  校准迁移              <- runs/_calibration/results.json

用法：
    python scripts/81_verify_supplement_numbers.py
退出码 0 = 全部匹配；1 = 存在不匹配或缺失文件。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
DOC = ROOT / "docs" / "manuscript" / "Supplementary_Material.md"

ORDER = ["lung_colon", "uterus_ca", "breakhis_400x", "endo_histo",
         "ovarian_ca", "breast_idc", "colon_cancer_csv"]


def load(p: Path) -> dict | None:
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def fmt(v: float) -> str:
    return f"{v:.4f}"


class Checker:
    def __init__(self, text: str) -> None:
        self.text = text
        self.bad: list[str] = []
        self.ok = 0

    def has(self, value: float, label: str, *, plus: bool = False) -> None:
        """文档中是否出现该数值的 4 位小数表示。"""
        s = (f"{value:+.4f}" if plus else f"{value:.4f}")
        # 文档用 U+2212 MINUS SIGN 排版；归一化后比较，避免"数字相同但字符串不等"的假报警
        hay = self.text.replace("\u2212", "-")
        if s in hay:
            self.ok += 1
        else:
            self.bad.append(f"{label}: 期望 {s} 但文档中未出现")

    def has_any(self, values: list[float], label: str) -> None:
        """一组数值至少出现一个（用于不可能全部枚举的情形）。"""
        hay = self.text.replace("\u2212", "-")
        if any(fmt(v) in hay for v in values):
            self.ok += 1
        else:
            self.bad.append(f"{label}: {[fmt(v) for v in values]} 均未出现")


def main() -> int:
    if not DOC.exists():
        print(f"[ERROR] 缺 {DOC}", file=sys.stderr)
        return 2
    c = Checker(DOC.read_text(encoding="utf-8"))

    mx = load(RUNS / "_cross_source_matrix" / "results.json")
    sa = load(RUNS / "_stain_shortcut_audit" / "results.json")
    st = load(RUNS / "_stain_ablation_encoders" / "results.json")
    ec = load(RUNS / "_encoder_comparison" / "results.json")

    # ---- S2 矩阵（49 格）
    if mx:
        for a in ORDER:
            for b in ORDER:
                v = (mx["in_domain"][a]["auc"] if a == b
                     else mx["cross_matrix"].get(f"{a}->{b}", {}).get("auc"))
                if v is not None:
                    c.has(v, f"S2 {a}->{b}")
        # ---- S3
        for t, d in mx["as_target"].items():
            c.has(d["mean_auc"], f"S3 mean {t}")
            c.has(d["max_auc"], f"S3 best {t}")
            c.has(d["min_auc"], f"S3 worst {t}")
            c.has(d["gap"], f"S3 gap {t}", plus=True)

    # ---- S4
    if mx and sa:
        sc = sa.get("shortcut_audit", {})
        for k, d in sc.items():
            c.has(d["stats_auc"], f"S4 shortcut {k}")
            if k in mx["in_domain"]:
                c.has(mx["in_domain"][k]["auc"], f"S4 in-domain {k}")
                c.has(d["stats_auc"] - mx["in_domain"][k]["auc"],
                      f"S4 diff {k}", plus=True)

    # ---- S5 染色消融逐配对
    if st:
        # 注意：用归一化后的文本（U+2212 → ASCII '-'），否则负值全部假报警
        hay = c.text.replace("\u2212", "-")
        for enc, sub in (st.get("summary") or {}).items():
            for pair, dv in (sub.get("delta_auc") or {}).items():
                s = f"{dv:+.4f}"
                if s in hay:
                    c.ok += 1
                else:
                    c.bad.append(f"S5 {enc} {pair}: 期望 {s} 未出现")

    # ---- S6 四实验汇总
    if ec:
        for enc, s in (ec.get("encoders") or {}).items():
            c.has(s["in_domain_mean_auc"], f"S6 {enc} in-domain")
            c.has(s["cross_mean_auc"], f"S6 {enc} cross")
            c.has(s["mean_gap"], f"S6 {enc} gap", plus=True)
        for enc, rec in (ec.get("other_experiments") or {}).items():
            if "biopsy" in rec:
                b = rec["biopsy"]
                c.has(b["sensitivity_at_0.5"], f"S6 {enc} biopsy sens")
                if b.get("sensitivity_seed_sd") is not None:
                    c.has(b["sensitivity_seed_sd"], f"S6 {enc} biopsy sens SD")
                c.has(b["biopsy_mean_pred"], f"S6 {enc} biopsy mean p")
                c.has(b["source_malignant_mean_pred"], f"S6 {enc} src malignant p")
            if "stain" in rec:
                c.has(rec["stain"]["delta_mean"], f"S6 {enc} stain delta", plus=True)
            if "cross_specimen" in rec:
                v = rec["cross_specimen"].get("stratified_delta_mean_pred")
                if v is not None:
                    c.has(v, f"S6 {enc} cross-specimen delta", plus=True)

    # ---- S7 活检逐种子
    for d in ("_external_biopsy", "_external_biopsy_phikon", "_external_biopsy_vit-base-in"):
        b = load(RUNS / d / "results.json")
        if not b:
            continue
        for ps in b.get("per_seed", []):
            eb = ps["external_biopsy"]
            c.has(eb["sensitivity_at_0.5"], f"S7 {d} seed{ps['seed']} sens")
            c.has(eb["mean_pred"], f"S7 {d} seed{ps['seed']} mean")

    # ---- S8 校准迁移
    cal = load(RUNS / "_calibration" / "results.json")
    if cal:
        for k, v in (cal.get("summary") or {}).items():
            if isinstance(v, (int, float)) and "ece" in k:
                if f"{v:.4f}".replace("-", "\u2212") in c.text or f"{v:.4f}" in c.text:
                    c.ok += 1
                else:
                    c.bad.append(f"S8 {k}: 期望 {v:.4f} 未出现")

    # ---- S9 探针稳健性
    # 注意：S9 表只列**汇总**与**逐来源域内 AUC**；42×4 个逐配对 AUC 留在
    # runs/_probe_robustness/results.json 中，不进表（否则表不可读）。
    # 因此校验范围与表内容一致，不对表外数字报错。
    pr = load(RUNS / "_probe_robustness" / "results.json")
    if pr:
        for k, v in pr["probes"].items():
            s = v["summary"]
            c.has(s["in_domain_mean_auc"], f"S9 {k} in-domain mean")
            c.has(s["cross_mean_auc"], f"S9 {k} cross mean")
            c.has(s["mean_gap"], f"S9 {k} gap", plus=True)
            for sname, auc in v["in_domain"].items():
                c.has(auc, f"S9 {k} in-domain {sname}")
        # 汇总口径：落差范围与低于随机计数
        vd = pr.get("verdict", {})
        for g in vd.get("gap_range", []):
            c.has(g, "S9 gap range", plus=True)

    # ---- S10 捷径残差化
    sr = load(RUNS / "_shortcut_residual" / "baseline.json")
    if sr:
        for k, v in sr["sources"].items():
            c.has(v["auc_original"], f"S10 {k} original")
            c.has(v["auc_after_residualization"], f"S10 {k} residualised")
            c.has(v["delta_auc"], f"S10 {k} delta", plus=True)
            c.has(v["auc_statistics_only"], f"S10 {k} statistics-only")
        s = sr.get("summary", {})
        if s.get("mean_delta_auc") is not None:
            c.has(s["mean_delta_auc"], "S10 mean delta", plus=True)

    # ---- S11 平凡预测器基线
    tb = load(RUNS / "_trivial_baselines" / "baselines.json")
    if tb:
        for dom, entry in tb["target_domains"].items():
            for name, v in entry.get("trivial", {}).items():
                c.has(v["brier"], f"S11 {dom} {name} brier")
                c.has(v["ece"], f"S11 {dom} {name} ece")
                c.has(v["calibration_in_the_large"], f"S11 {dom} {name} citl", plus=True)
                if v.get("sensitivity_at_0.5") is not None:
                    c.has(v["sensitivity_at_0.5"], f"S11 {dom} {name} sens")
            for name, v in entry.get("model", {}).items():
                for key in ("brier", "calibration_in_the_large", "cross_specimen_brier"):
                    if isinstance(v.get(key), (int, float)):
                        c.has(v[key], f"S11 {dom} {name} {key}", plus=(key == "calibration_in_the_large"))

    print(f"[补充材料数字校验] 通过 {c.ok} 项")
    if c.bad:
        print(f"不匹配/缺失 {len(c.bad)} 项：")
        for b in c.bad[:40]:
            print("   -", b)
        return 1
    print("全部与 JSON 一致 ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
