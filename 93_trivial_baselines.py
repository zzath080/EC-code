"""活检/跨标本类型目标域上的**平凡预测器**基线。

为什么必须做
------------
两个 WSI 级目标域都**没有阴性样本**（91 张活检全为恶性；TCGA `01Z` 诊断切片全为恶性）。
在这种队列上：

    * 敏感度 = 1 可由常数预测器 p ≡ 1.0 平凡达到；
    * Brier 与 ECE 同样被常数预测器平凡达到（p ≡ 1.0 时 Brier = ECE = 0）。

也就是说，**这两个指标在该队列上无法区分分类器与常数预测器**。若不主动写出这一点，
审稿人自己算出来会认为作者在回避；主动写出反而是支持本文主线的证据：
"现有评测协议不足以判断模型能否跨来源使用"。

本脚本把每个目标域的常数基线算全，供正文引用。所有数值可复算，不做任何手工转录。

用法：
    python scripts/93_trivial_baselines.py
输出：
    runs/_trivial_baselines/baselines.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "_trivial_baselines"


def ece_binary(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """与 31_calibration_gap_test.py 同实现（10 等宽分箱），保证可比。"""
    edges = np.linspace(0, 1, bins + 1)
    tot = len(y)
    v = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum():
            v += (m.sum() / tot) * abs(p[m].mean() - y[m].mean())
    return float(v)


def summarise(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    out = {
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "prevalence": round(float(y.mean()), 4),
        "mean_pred": round(float(p.mean()), 4),
        "brier": round(float(np.mean((p - y) ** 2)), 4),
        "ece": round(ece_binary(y, p), 4),
        "calibration_in_the_large": round(float(p.mean() - y.mean()), 4),
        "sensitivity_at_0.5": round(float((p[y == 1] >= thr).mean()), 4) if y.sum() else None,
    }
    # AUC 仅在两类都存在时可算
    if len(set(y.tolist())) == 2:
        from sklearn.metrics import roc_auc_score
        out["auc"] = round(float(roc_auc_score(y, p)), 4)
    else:
        out["auc"] = None
        out["auc_note"] = "single-class target: AUC undefined"
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    res: dict = {
        "note": ("两个 WSI 级目标域均**无阴性样本**，故敏感度与 Brier 被常数预测器平凡达到，"
                 "无法区分分类器与平凡解。这里列出全部常数基线供正文引用。"),
        "target_domains": {},
    }

    # ---- 目标域 1：ENDO-AID 活检（91 张，全恶性）
    bp = RUNS / "_external_biopsy" / "results.json"
    if bp.exists():
        j = json.loads(bp.read_text(encoding="utf-8"))
        n = j["target_domain"]["n_slides"]
        n_pos = int(j["aggregate"]["sensitivity_at_0.5"]["mean"] * 0)  # 占位，实际全恶性
        y = np.ones(n, dtype=int)          # 队列全为恶性（已在多处核实）
        entry: dict = {"cohort": "ENDO-AID biopsy", "all_malignant": True,
                       "trivial": {}, "model": {}}
        for name, p in {
            "constant_p_1.0": np.full(n, 1.0),
            "constant_p_0.5": np.full(n, 0.5),
            "constant_p_prevalence(=1.0)": np.full(n, float(y.mean())),
            "random_uniform": None,
        }.items():
            if p is None:
                rng = np.random.default_rng(0)
                p = rng.uniform(size=n)
            entry["trivial"][name] = summarise(y, p)
        # 三个编码器的实际表现
        for tag, d in (("phikon-v2", "_external_biopsy"),
                       ("phikon", "_external_biopsy_phikon"),
                       ("vit-base-in", "_external_biopsy_vit-base-in")):
            f = RUNS / d / "results.json"
            if not f.exists():
                continue
            jj = json.loads(f.read_text(encoding="utf-8"))
            agg = jj["aggregate"]
            entry["model"][tag] = {
                "sensitivity_at_0.5": agg["sensitivity_at_0.5"]["mean"],
                "sensitivity_seed_sd": agg["sensitivity_at_0.5"]["std"],
                "biopsy_mean_pred": agg["biopsy_mean_pred"]["mean"],
                # 目标域全为恶性 -> 真实阳性率 1.0 -> CITL = mean_pred - 1.0
                "calibration_in_the_large": round(agg["biopsy_mean_pred"]["mean"] - 1.0, 4),
                "brier": round(float((agg["biopsy_mean_pred"]["mean"] - 1.0) ** 2), 4),
                "source_val_auc": agg["source_val_auc"]["mean"],
            }
        res["target_domains"]["endoaid_biopsy"] = entry
        print(f"=== ENDO-AID 活检（n={n}，全为恶性）===")
        print(f"{'预测器':30s} {'敏感度@0.5':>10s} {'Brier':>8s} {'ECE':>8s} {'CITL':>8s}")
        for k, v in entry["trivial"].items():
            print(f"  {k:28s} {v['sensitivity_at_0.5']:10.4f} {v['brier']:8.4f} "
                  f"{v['ece']:8.4f} {v['calibration_in_the_large']:8.4f}")
        for k, v in entry["model"].items():
            print(f"  {k+' (本文模型)':28s} {v['sensitivity_at_0.5']:10.4f} {v['brier']:8.4f} "
                  f"{'n/a':>8s} {v['calibration_in_the_large']:8.4f}")
        print()

    # ---- 目标域 2：TCGA FFPE 诊断切片（全恶性）
    cp = RUNS / "_cross_specimen" / "calibration_gap.json"
    if cp.exists():
        j = json.loads(cp.read_text(encoding="utf-8"))
        st = j["stratified_by_label"]["malignant"]
        n = st["n_cross"]
        y = np.ones(n, dtype=int)
        entry = {"cohort": "TCGA-UCEC FFPE diagnostic (01Z)", "all_malignant": True,
                 "trivial": {}, "model": {}}
        for name, p in {"constant_p_1.0": np.full(n, 1.0),
                        "constant_p_0.5": np.full(n, 0.5)}.items():
            entry["trivial"][name] = summarise(y, p)
        entry["model"]["phikon-v2"] = {
            "mean_pred": st["mean_pred_cross"],
            "calibration_in_the_large": round(st["mean_pred_cross"] - 1.0, 4),
            "cross_specimen_ece": j["cross_domain"]["ece"],
            "cross_specimen_brier": j["cross_domain"]["brier"],
        }
        res["target_domains"]["tcga_01z_diagnostic"] = entry
        print(f"=== TCGA FFPE 诊断切片（n={n}，全为恶性）===")
        for k, v in entry["trivial"].items():
            print(f"  {k:28s} {'敏感度 n/a':>10s} {v['brier']:8.4f} {v['ece']:8.4f} "
                  f"{v['calibration_in_the_large']:8.4f}")
        m = entry["model"]["phikon-v2"]
        print(f"  {'phikon-v2 (本文模型)':28s} {'':>10s} {m['cross_specimen_brier']:8.4f} "
              f"{m['cross_specimen_ece']:8.4f} {m['calibration_in_the_large']:8.4f}")
        print()

    # ---- 域内（有良恶性，AUC 才有意义）作为对照
    ip = RUNS / "_cross_specimen" / "results.json"
    if ip.exists():
        j = json.loads(ip.read_text(encoding="utf-8"))
        ind = j["in_domain_tissue"]
        n = ind["n"]
        y = np.array([1] * (n // 2) + [0] * (n - n // 2))
        entry = {"cohort": "TCGA frozen tissue test (balanced)", "all_malignant": False,
                 "model": {"phikon-v2": {"auc": ind["auc"], "ece": ind["ece"],
                                         "brier": ind["brier"]}},
                 "trivial": {}}
        for name, p in {"constant_p_0.5": np.full(n, 0.5),
                        "constant_p_1.0": np.full(n, 1.0)}.items():
            entry["trivial"][name] = summarise(y, p)
        res["target_domains"]["tcga_frozen_test_balanced"] = entry
        print("=== 域内冰冻组织测试集（平衡 10/10，AUC 可用）===")
        print(f"  {'constant_p_0.5':28s} {'':>10s} {entry['trivial']['constant_p_0.5']['brier']:8.4f} "
              f"{entry['trivial']['constant_p_0.5']['ece']:8.4f}")
        print(f"  {'phikon-v2 (本文模型)':28s} {'':>10s} {ind['brier']:8.4f} {ind['ece']:8.4f}"
              f"   AUC={ind['auc']:.4f}")

    (OUT / "baselines.json").write_text(json.dumps(res, indent=2, ensure_ascii=False),
                                        encoding="utf-8")
    (OUT / "baselines.json").read_text(encoding="utf-8")
    print(f"\n-> {OUT/'baselines.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
