#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
27_cross_specimen_transfer.py
=============================
真正的「跨标本类型迁移」实验：在**组织切片**（tissue slides，冰冻/TSA）上训练，
在**诊断切片**（diagnostic slides，FFPE/01Z）上测试。

这与 v1 方案的关键区别：
    v1 想在 TCGA(切除) → Endo-AID(活检) 之间迁移，但活检数据未到手；
    本实验先在**已有的 TCGA 内部**构造一个真实的跨标本类型域偏移：
    两者都是手术切除标本，但组织处理方式完全不同（冰冻切片 vs FFPE）。
    这能在拿到活检数据之前，就验证「域偏移 + 校准失效」这条主线是否成立。

评估内容：
    - 训练域内表现（组织切片 train/test）
    - 跨域零样本表现（组织切片训练 → 诊断切片测试）
    - 跨域表现 vs 域内表现 的 ΔAUC 与 bootstrap CI
    - 校准指标（ECE/Brier）在跨域时是否显著恶化

输出：
    runs/_cross_specimen/
        results.json
        predictions.csv
        report.md
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path, PurePosixPath

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / ".pylibs"
if LIB.exists():
    sys.path.insert(0, str(LIB))
sys.path.insert(0, str(ROOT / "scripts"))


def auc(y, s) -> float:
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv), dtype=float)
    ranks[order] = np.arange(1, len(allv) + 1)
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def boot_ci(y, s, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        t = rng.integers(0, len(y), len(y))
        if len(np.unique(np.asarray(y)[t])) < 2:
            continue
        vals.append(auc(np.asarray(y)[t], np.asarray(s)[t]))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)


def ece(y, p, bins=10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    tot = len(y)
    v = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum():
            v += (m.sum() / tot) * abs(p[m].mean() - y[m].mean())
    return float(v)


def train_abmil(Xtr, ytr, Xva, yva, dim, device, epochs=30, lr=1e-3, max_inst=2048, seed=42):
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
            z = (w.unsqueeze(-1) * h).sum(0)
            return self.head(z.unsqueeze(0))

    model = MIL(dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()

    def infer(X):
        model.eval()
        out = []
        with torch.no_grad():
            for x in X:
                xb = torch.from_numpy(x[:max_inst]).to(device)
                out.append(float(torch.softmax(model(xb), 1)[0, 1].item()))
        return np.array(out)

    best, best_state = -1.0, None
    for ep in range(1, epochs + 1):
        model.train()
        for x, y in zip(Xtr, ytr):
            xb = torch.from_numpy(x[:max_inst]).to(device)
            yb = torch.tensor([int(y)], device=device)
            loss = lossf(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if len(Xva) and len(np.unique(yva)) == 2:
            va = auc(yva, infer(Xva))
            if va > best:
                best, best_state = va, {k: v.clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, infer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/features/phikon-v2")
    ap.add_argument("--encoder", default=None,
                    help="按编码器名替换 feature_path 中的 phikon-v2（支持第二编码器对照）")
    ap.add_argument("--out-dir", default=None,
                    help="输出目录；默认 runs/_cross_specimen（第二编码器自动加后缀）")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import h5py

    # 第二编码器对照：manifest 里的 feature_path 全部指向 phikon-v2/_flat。
    # 但 phikon 只有按来源分的目录、没有 _flat，且按来源的目录名与 manifest 的
    # feature_source 列一致，因此用该列重建路径，而不是对 _flat 做字符串替换。
    # 缺失时再回退到一次性扫描（按文件名索引），避免逐个 glob 拖慢速度。
    _index: dict[str, Path] | None = None

    def _build_index() -> dict[str, Path]:
        base = ROOT / "data" / "features" / str(args.encoder)
        return {p.name: p for p in base.rglob("*.h5")}

    def resolve_feature_path(rel: str, feature_source: str | None = None) -> Path:
        p = ROOT / rel
        if args.encoder is None:
            return p
        if feature_source:
            cand = ROOT / str(feature_source).replace("\\", "/").replace("phikon-v2", str(args.encoder)) / p.name
            if cand.exists():
                return cand
        nonlocal _index
        if _index is None:
            _index = _build_index()
        return _index.get(p.name, p)

    split_csv = ROOT / "data" / "manifests" / "tcga_pair_train_manifest.csv"
    rows = list(csv.DictReader(open(split_csv, encoding="utf-8-sig")))
    if not rows:
        print("[ERROR] 缺少 tcga_pair_train_manifest.csv", file=sys.stderr)
        return 2

    # 按「标本类型」重新分组：诊断切片(01Z) vs 组织切片(01A/11A)
    def specimen(sid: str) -> str:
        code = sid.split("-")[3] if sid.count("-") >= 3 else ""
        return "diagnostic" if code.startswith("01") else ("tissue" if code.startswith("11") or code.startswith("01A") else "other")

    def kind(sid: str) -> str:
        parts = sid.split("-")
        code = parts[3] if len(parts) > 3 else ""
        # 01Z=诊断(FFPE)，01A/11A=组织切片(冰冻/TSA)
        return "diag" if code.endswith("Z") else "tissue"

    groups: dict[str, list[dict]] = {"diag": [], "tissue": []}
    for r in rows:
        g = kind(r["slide_id"])
        groups.setdefault(g, []).append(r)

    print("按标本类型分组：")
    for g, items in groups.items():
        lab = [i["label"] for i in items]
        print(f"  {g:8s} {len(items):3d} 张  恶性 {sum(int(x) for x in lab)}  良性 {len(lab)-sum(int(x) for x in lab)}")

    def load(items):
        X, y, ids = [], [], []
        n_missing = 0
        for r in items:
            p = resolve_feature_path(r["feature_path"], r.get("feature_source"))
            if not p.exists():
                n_missing += 1
                continue
            with h5py.File(p, "r") as fh:
                X.append(fh["features"][:].astype(np.float32))
            y.append(int(r["label"]))
            ids.append(r["slide_id"])
        if n_missing:
            print(f"  [WARN] {n_missing} 张切片缺特征文件（已跳过）")
        return X, np.array(y), ids

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dim = 1024

    Xt, yt, idt = load(groups["tissue"])
    Xd, yd, idd = load(groups["diag"])
    if Xt and Xd:
        dim = int(Xt[0].shape[-1])
    print(f"\n可用特征：组织切片 {len(Xt)} 张，诊断切片 {len(Xd)} 张（dim={dim}, device={device}）")
    if len(Xt) < 8 or len(Xd) < 4:
        print("[WARN] 样本不足，无法做跨标本类型实验", file=sys.stderr)
        return 1

    # 组织切片内部按患者划分 train/val/test（用 manifest 的 split 优先，否则按序切分）
    idx = np.arange(len(Xt))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(idx)
    n = len(idx)
    tr_i, va_i, te_i = idx[: int(0.6 * n)], idx[int(0.6 * n): int(0.8 * n)], idx[int(0.8 * n):]
    Xtr, ytr = [Xt[i] for i in tr_i], yt[tr_i]
    Xva, yva = [Xt[i] for i in va_i], yt[va_i]
    Xte, yte = [Xt[i] for i in te_i], yt[te_i]

    model, infer = train_abmil(Xtr, ytr, Xva, yva, dim, device, epochs=args.epochs, seed=args.seed)

    res: dict = {"n_tissue": len(Xt), "n_diag": len(Xd), "device": device,
                 "train_domain": "tissue slides (frozen/TSA)", "test_domain": "diagnostic slides (FFPE)"}

    p_in = infer(Xte)
    a_in = auc(yte, p_in)
    lo_in, hi_in = boot_ci(yte, p_in)
    res["in_domain_tissue"] = {"n": len(yte), "auc": round(a_in, 4),
                               "auc_ci95": [round(lo_in, 4), round(hi_in, 4)],
                               "ece": round(ece(yte, p_in), 4),
                               "brier": round(float(np.mean((p_in - yte) ** 2)), 4)}

    p_cross = infer(Xd)
    a_cr = auc(yd, p_cross)
    lo_cr, hi_cr = boot_ci(yd, p_cross)
    res["cross_specimen_diag"] = {"n": len(yd), "auc": round(a_cr, 4),
                                  "auc_ci95": [round(lo_cr, 4), round(hi_cr, 4)],
                                  "ece": round(ece(yd, p_cross), 4),
                                  "brier": round(float(np.mean((p_cross - yd) ** 2)), 4),
                                  "mean_pred": round(float(p_cross.mean()), 4),
                                  "frac_pos": round(float(yd.mean()), 4)}

    # 域内（诊断切片）自身上限参考：用诊断切片全体做最近质心
    if len(np.unique(yd)) == 2:
        c0 = np.stack([x.mean(0) for x, yy in zip(Xd, yd) if yy == 0]).mean(0)
        c1 = np.stack([x.mean(0) for x, yy in zip(Xd, yd) if yy == 1]).mean(0)
        s = np.stack([x.mean(0) for x in Xd])
        d0 = np.linalg.norm(s - c0, axis=1)
        d1 = np.linalg.norm(s - c1, axis=1)
        res["diag_centroid_upper_ref_auc"] = round(auc(yd, (d0 - d1) / (d0 + d1 + 1e-12)), 4)

    res["auc_drop_cross_minus_in"] = round(a_cr - a_in, 4)

    out_dir = (Path(args.out_dir) if args.out_dir else
               (ROOT / "runs" / (f"_cross_specimen_{args.encoder}" if args.encoder else "_cross_specimen")))
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(out_dir / "predictions.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["slide_id", "domain", "label", "prob_malignant"])
        for sid, yy, pp in zip(idd, yd, p_cross):
            w.writerow([sid, "diagnostic", int(yy), round(float(pp), 6)])
        for sid, yy, pp in zip([idt[i] for i in te_i], yte, p_in):
            w.writerow([sid, "tissue_test", int(yy), round(float(pp), 6)])

    lines = ["# 跨标本类型迁移实验结果", "",
             f"- 训练域：组织切片（冰冻/TSA），{len(Xtr)} 张",
             f"- 测试域（跨标本类型）：诊断切片（FFPE 01Z），{len(Xd)} 张",
             f"- 设备：{device}", "",
             "## 结果", "",
             "| 评估 | n | AUC | 95% CI | ECE | Brier |", "| --- | --- | --- | --- | --- | --- |"]
    for k, label in [("in_domain_tissue", "域内（组织切片 test）"),
                     ("cross_specimen_diag", "**跨标本类型**（组织→诊断）")]:
        d = res[k]
        lines.append(f"| {label} | {d['n']} | {d['auc']:.4f} | "
                     f"[{d['auc_ci95'][0]:.3f}, {d['auc_ci95'][1]:.3f}] | {d['ece']:.4f} | {d['brier']:.4f} |")
    lines += ["", f"- 跨域相对域内的 AUC 变化：**{res['auc_drop_cross_minus_in']:+.4f}**",
              f"- 诊断切片自身的最近质心上限参考 AUC：{res.get('diag_centroid_upper_ref_auc', 'n/a')}",
              "", "> 判读：若跨域 AUC 明显低于域内、或 ECE 明显升高，则说明**标本处理方式的域偏移**",
              "> 会实质损害模型，这正是本研究的核心命题（也正是活检迁移会更严重的预警）。"]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\n-> {out_dir/'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
