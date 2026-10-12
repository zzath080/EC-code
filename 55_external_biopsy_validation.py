#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
55_external_biopsy_validation.py
================================
外部活检域验证（Plan B 的 B3 实验）：

    源域：TCGA-UCEC 手术切除冰冻组织切片（恶性 = 原发瘤，良性 = 实性正常组织）
    目标域：ENDO-AID 真实 Pipelle 活检切片（**全部为恶性**，无良性）

可报告的指标（单类别目标域下）：
    - 恶性召回率（sensitivity）@ 默认阈值 0.5 与 @ 高敏感度阈值
    - 平均预测概率 vs 目标域真实阳性率 → **calibration-in-the-large 偏差**
    - 预测分布（离散度）与源域对比
    - 每切片 patch 数与预测的关系（组织量是否影响判定）

不可报告：AUC、特异度、准确率（目标域无良性样本）。

用法：
    python scripts/55_external_biopsy_validation.py --epochs 30
    python scripts/55_external_biopsy_validation.py --seed 0 --seed 1 --seed 2   # 多种子重复
输出：
    runs/_external_biopsy/results.json
    runs/_external_biopsy/report.md
    runs/_external_biopsy/predictions.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / ".pylibs"
if LIB.exists():
    sys.path.insert(0, str(LIB))

DEFAULT_ENCODER = "phikon-v2"


def feat_root(encoder: str) -> Path:
    return ROOT / "data" / "features" / encoder


def out_dir_for(encoder: str) -> Path:
    return ROOT / "runs" / ("_external_biopsy" if encoder == DEFAULT_ENCODER
                            else f"_external_biopsy_{encoder}")


FEAT_ROOT = feat_root(DEFAULT_ENCODER)   # 兼容旧调用
SRC_COHORTS = ["tcga_normal", "tcga_tumor", "tcga_tumor_legacy", "tcga_01z"]
TGT_COHORT = "endoaid_biopsy"
OUT = ROOT / "runs" / "_external_biopsy"


def load_h5_dir(d: Path) -> dict[str, np.ndarray]:
    import h5py
    out = {}
    if not d.exists():
        return out
    for p in sorted(d.glob("*.h5")):
        with h5py.File(p, "r") as fh:
            out[p.stem] = fh["features"][:].astype(np.float32)
    return out


def specimen_kind_of(sid: str) -> str:
    parts = sid.split("-")
    code = parts[3] if len(parts) > 3 else ""
    if code.endswith("Z"):
        return "diag"
    if code.startswith("11"):
        return "normal_tissue"
    return "tissue"


def auc(y, s) -> float:
    y = np.asarray(y).astype(int); s = np.asarray(s, dtype=float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def train_abmil(Xtr, ytr, Xva, yva, dim, device, epochs, seed, max_inst=2048):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    hidden = 256

    class MIL(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(0.25))
            self.att_net = nn.Sequential(nn.Linear(d, hidden), nn.Tanh(), nn.Linear(hidden, 1))
            self.head = nn.Linear(hidden, 2)

        def forward(self, x):
            h = self.encoder(x)
            a = self.att_net(x).squeeze(-1)
            w = torch.softmax(a, dim=0)
            return self.head((w.unsqueeze(-1) * h).sum(0).unsqueeze(0))

    model = MIL(dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()

    def infer(X):
        model.eval(); out = []
        with torch.no_grad():
            for x in X:
                xb = torch.from_numpy(x[:max_inst]).to(device)
                out.append(float(torch.softmax(model(xb), 1)[0, 1].item()))
        return np.array(out)

    best, best_state = -1.0, None
    for _ in range(epochs):
        model.train()
        for x, y in zip(Xtr, ytr):
            xb = torch.from_numpy(x[:max_inst]).to(device)
            yb = torch.tensor([int(y)], device=device)
            loss = lossf(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        if len(Xva) and len(np.unique(yva)) == 2:
            a = auc(yva, infer(Xva))
            if a > best:
                best, best_state = a, {k: v.clone() for k, v in model.state_dict().items()}
    if best_state:
        model.load_state_dict(best_state)
    return model, infer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--max-instances", type=int, default=2048)
    ap.add_argument("--encoder", default=DEFAULT_ENCODER)
    args = ap.parse_args()

    feats = feat_root(args.encoder)
    out = out_dir_for(args.encoder)
    print(f"编码器 {args.encoder}；特征目录 {feats.relative_to(ROOT)}")
    # 源域：排除 01Z 诊断切片（保持"切除冰冻标本"的域定义一致）
    src_slides: dict[str, np.ndarray] = {}
    for c in SRC_COHORTS:
        for sid, X in load_h5_dir(feats / c).items():
            if specimen_kind_of(sid) == "normal_tissue" or sid.startswith(("TCGA",)) and not sid.split("-")[3].startswith("01Z"):
                src_slides.setdefault(sid, X)
    # 明确标签：11A = 良性，01A/01B = 恶性
    labeled = {}
    for sid, X in src_slides.items():
        parts = sid.split("-")
        code = parts[3] if len(parts) > 3 else ""
        if code.startswith("11"):
            labeled[sid] = (X, 0, "normal_tissue")
        elif code.startswith("01A") or code.startswith("01B"):
            labeled[sid] = (X, 1, "tumor_tissue")
    tgt = load_h5_dir(feats / TGT_COHORT)

    print(f"源域切片 {len(labeled)}（良性 {sum(1 for v in labeled.values() if v[1]==0)} / "
          f"恶性 {sum(1 for v in labeled.values() if v[1]==1)}）")
    print(f"目标域活检切片 {len(tgt)}（全部为恶性）")
    if len(labeled) < 20 or len(tgt) < 10:
        print("[WARN] 样本不足，无法完成实验", file=sys.stderr)
        return 1

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dim = next(iter(labeled.values()))[0].shape[1]

    ids = sorted(labeled)
    rng = np.random.default_rng(42)
    order = rng.permutation(len(ids))
    n = len(ids)
    tr_i = order[: int(0.7 * n)]
    va_i = order[int(0.7 * n): int(0.85 * n)]
    ids_tr = [ids[i] for i in tr_i]
    ids_va = [ids[i] for i in va_i]

    Xtr = [labeled[s][0] for s in ids_tr]; ytr = np.array([labeled[s][1] for s in ids_tr])
    Xva = [labeled[s][0] for s in ids_va]; yva = np.array([labeled[s][1] for s in ids_va])
    tgt_ids = sorted(tgt)
    Xtgt = [tgt[s] for s in tgt_ids]

    per_seed = []
    for seed in args.seeds:
        t0 = time.time()
        model, infer = train_abmil(Xtr, ytr, Xva, yva, dim, device, args.epochs, seed, args.max_instances)
        p_src_mal = infer([labeled[s][0] for s in ids if labeled[s][1] == 1])
        p_tgt = infer(Xtgt)
        rec = {
            "seed": seed,
            "source_val_auc": round(auc(yva, infer(Xva)), 4),
            "external_biopsy": {
                "n": len(p_tgt), "n_malignant": len(p_tgt),
                "sensitivity_at_0.5": round(float((p_tgt >= 0.5).mean()), 4),
                "mean_pred": round(float(p_tgt.mean()), 4),
                "std_pred": round(float(p_tgt.std()), 4),
                "min_pred": round(float(p_tgt.min()), 4),
                "max_pred": round(float(p_tgt.max()), 4),
                "n_below_0.5": int((p_tgt < 0.5).sum()),
                "n_below_0.1": int((p_tgt < 0.1).sum()),
            },
            "source_malignant_mean_pred": round(float(p_src_mal.mean()), 4),
            "source_malignant_std_pred": round(float(p_src_mal.std()), 4),
            "calibration_in_the_large_bias": round(float(p_tgt.mean() - 1.0), 4),
            "seconds": round(time.time() - t0, 1),
        }
        per_seed.append(rec)
        print(f"  seed {seed}: 源域val AUC={rec['source_val_auc']:.4f}  "
              f"活检召回@0.5={rec['external_biopsy']['sensitivity_at_0.5']:.4f}  "
              f"活检平均预测={rec['external_biopsy']['mean_pred']:.4f}  "
              f"源域恶性平均预测={rec['source_malignant_mean_pred']:.4f}")

    def agg(key_path: list[str]) -> dict:
        vals = []
        for r in per_seed:
            v = r
            for k in key_path:
                v = v[k]
            vals.append(v)
        a = np.array(vals, dtype=float)
        return {"mean": round(float(a.mean()), 4), "std": round(float(a.std()), 4),
                "min": round(float(a.min()), 4), "max": round(float(a.max()), 4)}

    summary = {
        "source_domain": {"cohorts": SRC_COHORTS, "n_slides": len(labeled),
                          "note": "TCGA 手术切除冰冻组织切片；恶性=01A/01B 原发瘤，良性=11A 实性正常组织"},
        "target_domain": {"cohort": TGT_COHORT, "n_slides": len(tgt),
                          "note": "ENDO-AID 真实 Pipelle 活检 WSI，全部为恶性"},
        "n_seeds": len(args.seeds),
        "aggregate": {
            "source_val_auc": agg(["source_val_auc"]),
            "sensitivity_at_0.5": agg(["external_biopsy", "sensitivity_at_0.5"]),
            "biopsy_mean_pred": agg(["external_biopsy", "mean_pred"]),
            "biopsy_std_pred": agg(["external_biopsy", "std_pred"]),
            "source_malignant_mean_pred": agg(["source_malignant_mean_pred"]),
        },
        "per_seed": per_seed,
        "limits": [
            "目标域全部为恶性，故 **AUC / 特异度 / 准确率均不可报告**；",
            "只有一个外部活检数据集（91 张），无第二个活检域可交叉验证；",
            "源域与目标域在标本处理、扫描仪、取材方式上均不同，无法分离单一因素。",
        ],
    }

    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # 用第一个种子写逐切片预测
    model, infer = train_abmil(Xtr, ytr, Xva, yva, dim, device, args.epochs, args.seeds[0], args.max_instances)
    p_tgt = infer(Xtgt)
    with open(out / "predictions.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["slide_id", "domain", "label", "prob_malignant", "n_patches"])
        for sid, p in zip(tgt_ids, p_tgt):
            w.writerow([sid, "external_biopsy", 1, round(float(p), 6), int(tgt[sid].shape[0])])

    a = summary["aggregate"]
    lines = [
        "# 外部活检域验证结果",
        "",
        "## 设计",
        "",
        f"- **源域**：TCGA 手术切除冰冻组织切片，{len(labeled)} 张（恶性 01A/01B，良性 11A）",
        f"- **目标域**：ENDO-AID 真实 Pipelle 活检 WSI，**{len(tgt)} 张，全部为恶性**",
        f"- 模型：Phikon-v2 冻结特征 + ABMIL；{len(args.seeds)} 个随机种子重复",
        "",
        "## 结果（多种子均值 ± 标准差）",
        "",
        "| 指标 | 均值 | 标准差 | 范围 |",
        "| --- | --- | --- | --- |",
        f"| 源域验证集 AUC | {a['source_val_auc']['mean']:.4f} | {a['source_val_auc']['std']:.4f} | "
        f"[{a['source_val_auc']['min']:.4f}, {a['source_val_auc']['max']:.4f}] |",
        f"| **活检恶性召回率 @0.5** | **{a['sensitivity_at_0.5']['mean']:.4f}** | "
        f"{a['sensitivity_at_0.5']['std']:.4f} | [{a['sensitivity_at_0.5']['min']:.4f}, "
        f"{a['sensitivity_at_0.5']['max']:.4f}] |",
        f"| 活检平均预测概率 | {a['biopsy_mean_pred']['mean']:.4f} | {a['biopsy_mean_pred']['std']:.4f} | "
        f"[{a['biopsy_mean_pred']['min']:.4f}, {a['biopsy_mean_pred']['max']:.4f}] |",
        f"| 活检预测离散度 | {a['biopsy_std_pred']['mean']:.4f} | {a['biopsy_std_pred']['std']:.4f} | — |",
        f"| 源域恶性平均预测 | {a['source_malignant_mean_pred']['mean']:.4f} | "
        f"{a['source_malignant_mean_pred']['std']:.4f} | — |",
        "",
        "## 判读",
        "",
        f"- 活检召回率均值 **{a['sensitivity_at_0.5']['mean']:.4f}**："
        f"{'表明模型在真实活检上基本能识别恶性' if a['sensitivity_at_0.5']['mean'] > 0.8 else '表明模型在真实活检上漏诊明显，域偏移严重'}；",
        f"- 活检平均预测 **{a['biopsy_mean_pred']['mean']:.4f}** vs 源域恶性 **{a['source_malignant_mean_pred']['mean']:.4f}**："
        f"差值 {a['biopsy_mean_pred']['mean'] - a['source_malignant_mean_pred']['mean']:+.4f}"
        f"（{'跨域置信度下降' if a['biopsy_mean_pred']['mean'] < a['source_malignant_mean_pred']['mean'] else '跨域置信度未下降'}）；",
        f"- 活检预测离散度 {a['biopsy_std_pred']['mean']:.4f} vs 源域 "
        f"{per_seed[0]['source_malignant_std_pred']:.4f}：反映判定稳定性。",
        "",
        "## 必须写入局限",
        "",
    ] + [f"{i}. {l}" for i, l in enumerate(summary["limits"], 1)]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(summary["aggregate"], indent=2, ensure_ascii=False))
    print(f"\n-> {out/'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
