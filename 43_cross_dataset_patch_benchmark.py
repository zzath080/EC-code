#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
43_cross_dataset_patch_benchmark.py
===================================
跨数据集 patch 级基准：在多个子宫内膜/子宫良恶性数据集上做
(a) 数据集内 5 折交叉验证，与
(b) 跨数据集零样本迁移（train on A → test on B）。

为什么这个实验有价值：
    「域偏移是否损害模型」这一核心命题
    可以在 patch 级、用**完全开放的数据**独立验证——而且跨数据集迁移是比
    随机划分更严格的域偏移检验（不同医院、不同扫描仪、不同染色流程）。

数据集：
    endo_histo : samihamuntahamahin/endometrial-histopathology (NE/EH/EP/EA，良 2767 / 恶 535)
    uterus_ca  : samihamuntahamahin/uterus-cancer (non-tumour 3334 / tumour 3000)

用法：
    python scripts/43_cross_dataset_patch_benchmark.py --extract
    python scripts/43_cross_dataset_patch_benchmark.py --eval
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

MODEL = ROOT / "models" / "phikon-v2"
FEAT = ROOT / "data" / "features" / "phikon-v2" / "patch_multidataset"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

DATASETS = {
    "endo_histo": {
        "root": ROOT / "data" / "raw" / "endometrial_patch" / "extracted" / "HIENet",
        "malignant": {"EA"},
        "benign": {"NE", "EH", "EP"},
        "note": "子宫内膜：正常/增生/息肉/腺癌",
    },
    "uterus_ca": {
        # 注意：40_verify_patch_dataset.py 的默认解压落点是 endometrial_patch/extracted，
        # 因此 Uterus-Cancer 也在该目录下（已实测确认）。
        "root": ROOT / "data" / "raw" / "endometrial_patch" / "extracted" / "Uterus-Cancer",
        "malignant": {"tumour"},
        "benign": {"non-tumour"},
        "note": "子宫：肿瘤 vs 非肿瘤",
    },
}


def list_images(spec: dict) -> list[tuple[Path, int, str]]:
    root: Path = spec["root"]
    out = []
    if not root.exists():
        return out
    for cls_dir in sorted(root.iterdir()):
        if not cls_dir.is_dir():
            continue
        name = cls_dir.name
        if name in spec["malignant"]:
            lab = 1
        elif name in spec["benign"]:
            lab = 0
        else:
            continue
        for p in sorted(cls_dir.glob("*")):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
                out.append((p, lab, name))
    return out


def cmd_extract(batch: int) -> int:
    import torch
    from PIL import Image
    from transformers import AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(str(MODEL)).eval().to(device)
    FEAT.mkdir(parents=True, exist_ok=True)

    for name, spec in DATASETS.items():
        items = list_images(spec)
        if not items:
            print(f"[{name}] 无图像，跳过（检查 {spec['root']}）")
            continue
        print(f"[{name}] {len(items)} 张，{spec['note']}")
        feats, labels, paths = [], [], []
        t0 = time.time()
        with torch.no_grad():
            for i in range(0, len(items), batch):
                chunk = items[i:i + batch]
                arrs = []
                for p, _, _ in chunk:
                    im = Image.open(p).convert("RGB").resize((224, 224), Image.BILINEAR)
                    arrs.append(np.asarray(im, dtype=np.uint8))
                x = torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2).float().div_(255.0)
                x = (x - torch.tensor(MEAN).view(1, 3, 1, 1)) / torch.tensor(STD).view(1, 3, 1, 1)
                out = model(x.to(device))
                emb = out.last_hidden_state[:, 0] if hasattr(out, "last_hidden_state") else out[0][:, 0]
                feats.append(emb.float().cpu().numpy())
                labels.extend([l for _, l, _ in chunk])
                paths.extend([str(p) for p, _, _ in chunk])
        X = np.concatenate(feats, axis=0).astype(np.float32)
        np.savez_compressed(FEAT / f"{name}.npz", features=X, labels=np.array(labels),
                            paths=np.array(paths))
        print(f"  -> {X.shape}  {time.time()-t0:.0f}s")

    return 0


def cmd_eval(seed: int) -> int:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.model_selection import GroupKFold

    data = {}
    for name in DATASETS:
        p = FEAT / f"{name}.npz"
        if p.exists():
            d = np.load(p, allow_pickle=True)
            data[name] = (d["features"], d["labels"].astype(int), d["paths"].astype(str))
            print(f"{name}: {len(d['labels'])} 张，阳性 {int(d['labels'].sum())}")
    if len(data) < 2:
        print("[WARN] 少于两个数据集，无法做跨数据集实验", file=sys.stderr)
        return 1

    res: dict = {"datasets": {}, "in_domain": {}, "cross_domain": {}}

    for name, (X, y, paths) in data.items():
        groups = np.array([Path(p).stem for p in paths])
        oof = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(X[tr], y[tr])
            oof[te] = clf.predict_proba(X[te])[:, 1]
        auc = float(roc_auc_score(y, oof))
        ap = float(average_precision_score(y, oof))
        res["in_domain"][name] = {"n": len(y), "n_pos": int(y.sum()), "auc": round(auc, 4),
                                  "average_precision": round(ap, 4),
                                  "mean_pred_pos": round(float(oof[y == 1].mean()), 4),
                                  "mean_pred_neg": round(float(oof[y == 0].mean()), 4)}
        res["datasets"][name] = {"n": len(y), "n_pos": int(y.sum()), "n_neg": int((1 - y).sum())}
        print(f"域内 {name}: AUC={auc:.4f} AP={ap:.4f}")

    # 跨数据集：train on A (全量) -> test on B (全量)
    names = list(data)
    for src in names:
        Xa, ya, _ = data[src]
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(Xa, ya)
        for dst in names:
            if dst == src:
                continue
            Xb, yb, _ = data[dst]
            if len(np.unique(yb)) < 2:
                continue
            p = clf.predict_proba(Xb)[:, 1]
            auc = float(roc_auc_score(yb, p))
            ap = float(average_precision_score(yb, p))
            res["cross_domain"][f"{src}->{dst}"] = {
                "auc": round(auc, 4), "average_precision": round(ap, 4),
                "mean_pred_pos": round(float(p[yb == 1].mean()), 4),
                "mean_pred_neg": round(float(p[yb == 0].mean()), 4),
                "n_test": int(len(yb)),
                # 关键校准指标：源域训练出的模型在目标域上的「预测偏移」
                "pred_shift_pos": round(float(p[yb == 1].mean() - clf.predict_proba(Xa)[ya == 1].mean()), 4),
            }
            print(f"跨数据集 {src} -> {dst}: AUC={auc:.4f} AP={ap:.4f} "
                  f"阳性均值={p[yb==1].mean():.3f} 阴性均值={p[yb==0].mean():.3f}")

    out = ROOT / "runs" / "_cross_dataset_patch"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["# 跨数据集 patch 级基准", "",
             "## 数据集", "", "| 数据集 | n | 阳性 | 阴性 |", "| --- | --- | --- | --- |"]
    for k, v in res["datasets"].items():
        lines.append(f"| {k} | {v['n']} | {v['n_pos']} | {v['n_neg']} |")
    lines += ["", "## 数据集内（5 折 GroupKFold）", "",
              "| 数据集 | n | AUC | AP | 恶性平均预测 | 良性平均预测 |", "| --- | --- | --- | --- | --- | --- |"]
    for k, v in res["in_domain"].items():
        lines.append(f"| {k} | {v['n']} | {v['auc']:.4f} | {v['average_precision']:.4f} | "
                     f"{v['mean_pred_pos']:.4f} | {v['mean_pred_neg']:.4f} |")
    if res["cross_domain"]:
        lines += ["", "## 跨数据集零样本迁移（train A → test B）", "",
                  "| 方向 | n | AUC | AP | 恶性平均预测 | 良性平均预测 | Δ恶性预测 |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        for k, v in res["cross_domain"].items():
            lines.append(f"| {k} | {v['n_test']} | {v['auc']:.4f} | {v['average_precision']:.4f} | "
                         f"{v['mean_pred_pos']:.4f} | {v['mean_pred_neg']:.4f} | {v['pred_shift_pos']:+.4f} |")
        lines += ["", "> 判读：若跨数据集 AUC 明显低于域内、或恶性平均预测显著下降（Δ 为负），",
                  "> 则说明**不同数据来源之间的域偏移确实损害了模型**——这与活检 WSI 任务的核心命题一致，",
                  "> 且该结论完全建立在公开可直接获取的数据上。"]
    lines += ["", "## 局限", "",
              "- patch 级，不代表 WSI 级性能；",
              "- 各数据集均**无患者标识**，只能按图像编号分组，无法完全排除同患者图像跨折；",
              "- 数据集来源、扫描仪与染色信息不完整，无法做混杂因素归因。"]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\n-> {out/'report.md'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if not (args.extract or args.eval):
        args.extract = args.eval = True
    if args.extract:
        cmd_extract(args.batch)
    if args.eval:
        cmd_eval(args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
