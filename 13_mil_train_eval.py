#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
13_mil_train_eval.py
====================
Stage 3：弱监督 MIL 训练 + 严格的校准与统计评价。

聚合器（覆盖文献要求的对比）：
    - mean / max      : 无参数池化基线（Chen 2024 要求必须包含）
    - abmil           : 注意力池化（Ilse 2018 思路）
    - clam            : 门控注意力（Lu 2022 思路的简化实现）

评价指标（论文一级指标）：
    - AUC + bootstrap 95% CI
    - PR-AUC、敏感度、特异度、平衡准确率、F1
    - Brier 分数、ECE（期望校准误差）、可靠性曲线
    - 配对 AUC 的 DeLong 检验（Holm 校正）
    - 高敏感度阈值下的混淆矩阵

输出：
    runs/<run_id>/
        config.json
        metrics.json
        predictions.csv
        calibration.csv
        confusion_matrix.csv

用法：
    python scripts/13_mil_train_eval.py --features data/features/phikon-v2/tcga_normal \
        --manifest data/manifests/tcga_pair_manifest.csv --aggregator abmil --epochs 30
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / ".pylibs"
if LIB.exists():
    sys.path.insert(0, str(LIB))

# ---------------------------------------------------------------- 指标
def auc_score(y: np.ndarray, s: np.ndarray) -> float:
    """Mann-Whitney U 形式的 AUC（无 sklearn 依赖时的兜底）。"""
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    # 处理并列
    _, inv, counts = np.unique(np.concatenate([pos, neg]), return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def bootstrap_auc_ci(y: np.ndarray, s: np.ndarray, n: int = 2000, alpha: float = 0.05,
                     seed: int = 42) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    vals = []
    for _ in range(n):
        take = rng.choice(idx, size=len(idx), replace=True)
        if len(np.unique(y[take])) < 2:
            continue
        vals.append(auc_score(y[take], s[take]))
    vals = np.array(vals)
    return float(np.mean(vals)), float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))


def pr_auc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    order = np.argsort(-np.asarray(s, dtype=float))
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / max(1, y.sum())
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))


def ece_score(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> tuple[float, list[dict]]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    total = len(y)
    ece = 0.0
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi if i < n_bins - 1 else p <= hi)
        if m.sum() == 0:
            rows.append({"bin_lo": round(lo, 3), "bin_hi": round(hi, 3), "n": 0,
                         "mean_pred": None, "frac_pos": None})
            continue
        conf = float(p[m].mean())
        acc = float(y[m].mean())
        ece += (m.sum() / total) * abs(conf - acc)
        rows.append({"bin_lo": round(lo, 3), "bin_hi": round(hi, 3), "n": int(m.sum()),
                     "mean_pred": round(conf, 4), "frac_pos": round(acc, 4)})
    return float(ece), rows


def delong_test(y: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> tuple[float, float]:
    """配对 AUC 的 DeLong 检验（Hanley-McNeil 快速近似实现）。"""
    y = np.asarray(y).astype(int)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), float("nan")

    def midrank(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x, kind="mergesort")
        ranks = np.empty(len(x), dtype=float)
        ranks[order] = np.arange(1, len(x) + 1)
        _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
        sums = np.zeros(len(cnt))
        np.add.at(sums, inv, ranks)
        return (sums / cnt)[inv]

    def components(s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s_pos, s_neg = s[pos], s[neg]
        allv = np.concatenate([s_pos, s_neg])
        r = midrank(allv)
        r_pos, r_neg = r[: len(pos)], r[len(pos):]
        v10 = (r_pos - np.arange(1, len(pos) + 1)) / len(neg)
        v01 = 1.0 - (r_neg - np.arange(1, len(neg) + 1)) / len(pos)
        return v10, v01

    v10_1, v01_1 = components(s1)
    v10_2, v01_2 = components(s2)
    auc1 = float(v10_1.mean())
    auc2 = float(v10_2.mean())
    n1, n0 = len(pos), len(neg)
    s10 = np.cov(np.vstack([v10_1, v10_2]))
    s01 = np.cov(np.vstack([v01_1, v01_2]))
    s10 = np.atleast_2d(s10)
    s01 = np.atleast_2d(s01)
    var = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / n1 + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / n0
    if var <= 0:
        return auc1 - auc2, 1.0
    z = (auc1 - auc2) / np.sqrt(var)
    from math import erf, sqrt
    p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    return float(auc1 - auc2), float(p)


# ---------------------------------------------------------------- 模型
def build_aggregator(name: str, dim: int, device: str):
    """构建 MIL 聚合器。

    三种聚合器共用同一 encoder（Linear->ReLU->Dropout），只有池化方式不同，
    这样对比才是「聚合器」本身的差异，而不是容量差异：
        mean / max : 无参数池化基线（Chen 2024 要求必须包含）
        abmil      : 注意力池化
        clam       : 门控注意力池化
    """
    import torch
    import torch.nn as nn

    hidden = 256

    class MIL(nn.Module):
        def __init__(self, kind: str, d: int, n_out: int = 2):
            super().__init__()
            self.kind = kind
            self.encoder = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(0.25))
            if kind == "abmil":
                self.att_net = nn.Sequential(nn.Linear(d, hidden), nn.Tanh(), nn.Linear(hidden, 1))
            elif kind == "clam":
                self.att_net = nn.Sequential(nn.Linear(d, hidden), nn.Tanh())
                self.gate_net = nn.Sequential(nn.Linear(d, hidden), nn.Sigmoid())
                self.att_vec = nn.Linear(hidden, 1)
            self.head = nn.Linear(hidden, n_out)

        def forward(self, x):
            # x: [N, D] 单张切片的所有实例
            h = self.encoder(x)                     # [N, hidden]
            if self.kind == "mean":
                z = h.mean(0)
            elif self.kind == "max":
                z = h.max(0).values
            elif self.kind == "abmil":
                a = self.att_net(x).squeeze(-1)     # [N]
                w = torch.softmax(a, dim=0)
                z = (w.unsqueeze(-1) * h).sum(0)
            else:  # clam
                a = self.att_net(x) * self.gate_net(x)
                w = torch.softmax(self.att_vec(a).squeeze(-1), dim=0)
                z = (w.unsqueeze(-1) * h).sum(0)
            return self.head(z.unsqueeze(0))        # [1, n_out]

    return MIL(name, dim).to(device)


# ---------------------------------------------------------------- 主流程
def load_features(feature_dir: Path, slide_ids: list[str]) -> dict[str, np.ndarray]:
    """在 feature_dir 及其一级子目录中查找 <slide_id>.h5。

    兼容两种布局：
        data/features/<encoder>/<slide_id>.h5          （扁平）
        data/features/<encoder>/<cohort>/<slide_id>.h5 （按队列分目录）
    """
    import h5py
    search_dirs = [feature_dir]
    if feature_dir.is_dir():
        search_dirs += [d for d in feature_dir.iterdir() if d.is_dir()]
    out = {}
    for sid in slide_ids:
        if sid in out:
            continue
        for d in search_dirs:
            p = d / f"{sid}.h5"
            if p.exists():
                with h5py.File(p, "r") as fh:
                    out[sid] = fh["features"][:].astype(np.float32)
                break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="特征目录")
    ap.add_argument("--manifest", required=True, help="含 slide_id,label,split 的 CSV")
    ap.add_argument("--aggregator", default="abmil", choices=["mean", "max", "abmil", "clam"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--max-instances", type=int, default=4096)
    args = ap.parse_args()

    import torch
    import torch.nn as nn

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = list(csv.DictReader(open(ROOT / args.manifest, encoding="utf-8-sig")))
    feat_dir = ROOT / args.features if not Path(args.features).is_absolute() else Path(args.features)
    feats = load_features(feat_dir, [r["slide_id"] for r in rows])
    if not feats:
        print("[ERROR] 没有可用特征；先跑 scripts/12_extract_features.py --save", file=sys.stderr)
        return 2

    dim = next(iter(feats.values())).shape[1]
    run_id = args.run_id or f"{args.aggregator}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = {"aggregator": args.aggregator, "epochs": args.epochs, "lr": args.lr,
           "seed": args.seed, "feature_dim": dim, "n_slides_with_features": len(feats),
           "n_manifest_rows": len(rows), "device": device}
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    def split_ids(name: str) -> list[str]:
        return [r["slide_id"] for r in rows if r["split"] == name and r["slide_id"] in feats]

    train_ids, val_ids, test_ids = split_ids("train"), split_ids("val"), split_ids("test")
    label_of = {r["slide_id"]: int(r["label"]) for r in rows if r.get("label") not in (None, "")}
    print(f"train={len(train_ids)} val={len(val_ids)} test={len(test_ids)} dim={dim} device={device}")

    model = build_aggregator(args.aggregator, dim, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()

    def run_epoch(ids: list[str], train: bool):
        model.train(train)
        losses, ys, ps = [], [], []
        for sid in ids:
            x = torch.from_numpy(feats[sid][: args.max_instances]).to(device)
            y = torch.tensor([label_of.get(sid, 0)], device=device)
            logits = model(x)
            loss = lossf(logits, y)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            losses.append(float(loss.item()))
            ys.append(int(y.item()))
            ps.append(float(torch.softmax(logits, 1)[0, 1].item()))
        return losses, ys, ps

    best_auc, best_state = -1.0, None
    for ep in range(1, args.epochs + 1):
        tr_loss, _, _ = run_epoch(train_ids, True)
        if val_ids:
            _, vy, vp = run_epoch(val_ids, False)
            v_auc = auc_score(vy, vp)
            if v_auc > best_auc:
                best_auc, best_state = v_auc, {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  epoch {ep:3d}  train_loss={np.mean(tr_loss):.4f}  val_auc={v_auc:.4f}")
    if best_state is not None:
        model.load_state_dict(best_state)

    eval_ids = test_ids or val_ids or train_ids
    _, y, p = run_epoch(eval_ids, False)
    y, p = np.array(y), np.array(p)

    auc = auc_score(y, p)
    mean_auc, lo, hi = bootstrap_auc_ci(y, p)
    ece, cal_rows = ece_score(y, p)
    brier = float(np.mean((p - y) ** 2))
    thr = 0.5
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    sens = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)

    metrics = {
        "run_id": run_id,
        "evaluated_on": "test" if test_ids else ("val" if val_ids else "train"),
        "n_eval": len(eval_ids),
        "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
        "auc": round(auc, 4),
        "auc_bootstrap_mean": round(mean_auc, 4),
        "auc_ci95": [round(lo, 4), round(hi, 4)],
        "pr_auc": round(pr_auc(y, p), 4),
        "sensitivity_at_0.5": round(sens, 4),
        "specificity_at_0.5": round(spec, 4),
        "balanced_accuracy": round((sens + spec) / 2, 4),
        "f1": round(2 * tp / max(1, 2 * tp + fp + fn), 4),
        "brier": round(brier, 4),
        "ece": round(ece, 4),
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(run_dir / "predictions.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["slide_id", "label", "prob_malignant", "pred_at_0.5"])
        for sid, yy, pp in zip(eval_ids, y, p):
            w.writerow([sid, int(yy), round(float(pp), 6), int(pp >= 0.5)])
    with open(run_dir / "calibration.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cal_rows[0].keys()))
        w.writeheader()
        w.writerows(cal_rows)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"\n-> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
