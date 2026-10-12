#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
41_patch_benchmark.py
=====================
在 patch 级子宫内膜数据集（HIENet：NE/EH/EP/EA 四类）上做基础模型基准。

这是本项目的子宫内膜 patch 级良恶性数据，用于：
    1. 验证 Phikon-v2 特征在子宫内膜组织上的可分性；
    2. 给出「良性 vs 恶性」的参考性能（作为活检 WSI 任务的下界参考）；
    3. 零成本 sanity check，不占用 WSI 下载与解码时间。

类别（HIENet 原命名）：
    NE = Normal Endometrium（正常子宫内膜）
    EH = Endometrial Hyperplasia（子宫内膜增生）
    EP = Endometrial Polyp（子宫内膜息肉）
    EA = Endometrial Adenocarcinoma（子宫内膜腺癌）

用法：
    python scripts/41_patch_benchmark.py --extract          # 提特征
    python scripts/41_patch_benchmark.py --eval             # 训练+评估
    python scripts/41_patch_benchmark.py --extract --eval
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / ".pylibs"
if LIB.exists():
    sys.path.insert(0, str(LIB))

SRC = ROOT / "data" / "raw" / "endometrial_patch" / "extracted" / "HIENet"
FEAT = ROOT / "data" / "features" / "phikon-v2" / "endometrial_patch"
MODEL = ROOT / "models" / "phikon-v2"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
BENIGN = {"NE", "EH", "EP"}
MALIGNANT = {"EA"}


def list_images() -> list[tuple[Path, str]]:
    out = []
    for cls_dir in sorted(SRC.iterdir()):
        if not cls_dir.is_dir():
            continue
        for p in sorted(cls_dir.glob("*")):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
                out.append((p, cls_dir.name))
    return out


def cmd_extract(batch: int, limit: int | None) -> int:
    import torch
    from PIL import Image
    from transformers import AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(str(MODEL)).eval().to(device)
    items = list_images()
    if limit:
        random.Random(0).shuffle(items)
        items = items[:limit]
    print(f"图像 {len(items)} 张，设备 {device}")

    FEAT.mkdir(parents=True, exist_ok=True)
    feats, labels, names = [], [], []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(items), batch):
            chunk = items[i:i + batch]
            arrs = []
            for p, _ in chunk:
                im = Image.open(p).convert("RGB").resize((224, 224), Image.BILINEAR)
                arrs.append(np.asarray(im, dtype=np.uint8))
            x = torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2).float().div_(255.0)
            x = (x - torch.tensor(MEAN).view(1, 3, 1, 1)) / torch.tensor(STD).view(1, 3, 1, 1)
            out = model(x.to(device))
            emb = out.last_hidden_state[:, 0] if hasattr(out, "last_hidden_state") else out[0][:, 0]
            feats.append(emb.float().cpu().numpy())
            labels.extend([c for _, c in chunk])
            names.extend([str(p.relative_to(SRC)) for p, _ in chunk])
            if (i // batch) % 20 == 0:
                print(f"  {min(i+batch, len(items))}/{len(items)}  {time.time()-t0:.0f}s", flush=True)

    X = np.concatenate(feats, axis=0).astype(np.float32)
    np.savez_compressed(FEAT / "patch_features.npz", features=X,
                        labels=np.array(labels), names=np.array(names))
    print(f"特征 {X.shape} -> {FEAT/'patch_features.npz'}  ({time.time()-t0:.0f}s)")
    return 0


def cmd_eval(seed: int) -> int:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
    from sklearn.model_selection import GroupKFold

    d = np.load(FEAT / "patch_features.npz", allow_pickle=True)
    X, labels, names = d["features"], d["labels"].astype(str), d["names"].astype(str)
    print("类别分布：", dict(Counter(labels)))

    # 二分类：良性(NE/EH/EP) vs 恶性(EA)
    y = np.array([1 if l in MALIGNANT else 0 for l in labels])
    # 患者/切片级分组：文件名形如 HIENet/EA/1509779.JPG，用编号做组以避免同源泄漏
    groups = np.array([Path(n).stem for n in names])

    res: dict = {"n": len(y), "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
                 "class_distribution": dict(Counter(labels))}

    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(y))
    for tr, te in gkf.split(X, y, groups):
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    res["binary_benign_vs_malignant"] = {
        "auc": round(float(roc_auc_score(y, oof)), 4),
        "average_precision": round(float(average_precision_score(y, oof)), 4),
        "confusion_at_0.5": confusion_matrix(y, (oof >= 0.5).astype(int)).tolist(),
        "note": "5 折 GroupKFold（按图像编号分组），逻辑回归线性探针",
    }

    # 四分类
    classes = sorted(set(labels))
    cls_idx = {c: i for i, c in enumerate(classes)}
    y4 = np.array([cls_idx[l] for l in labels])
    oof4 = np.zeros((len(y4), len(classes)))
    for tr, te in gkf.split(X, y4, groups):
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X[tr], y4[tr])
        oof4[te] = clf.predict_proba(X[te])
    acc = float((oof4.argmax(1) == y4).mean())
    res["four_class"] = {"classes": classes, "accuracy": round(acc, 4),
                         "note": "5 折 GroupKFold，随机基线 = %.4f" % (1 / len(classes))}

    out = ROOT / "runs" / "_patch_benchmark"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(out / "predictions.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "label", "prob_malignant"])
        for n, l, p in zip(names, labels, oof):
            w.writerow([n, l, round(float(p), 6)])

    print(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\n-> {out/'results.json'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if not (args.extract or args.eval):
        args.extract = args.eval = True
    if args.extract:
        cmd_extract(args.batch, args.limit)
    if args.eval:
        cmd_eval(args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
