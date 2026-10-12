#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
44_attribute_cross_dataset_failure.py
=====================================
对「跨数据集迁移失败」做归因，区分三种可能原因：
    H1 真实的域偏移（不同医院/扫描仪/染色）
    H2 类别语义不同（"non-tumour" ≠ "良性子宫内膜"）
    H3 数据集伪影（图像尺寸/文件大小等捷径特征）

做法：
    1. 检查每个数据集的图像尺寸与文件大小分布——若同数据集内类别间存在显著差异，
       则 H3 成立（模型可能学到尺寸/压缩伪影而非组织形态）；
    2. 用「仅图像元数据」训练分类器，测其 AUC——这是 H3 的直接检验（应当接近 0.5）；
    3. 比较两数据集的整体特征分布（均值嵌入的余弦相似度、类间可分性），评估 H1/H2。

输出：
    runs/_cross_dataset_patch/attribution.json
    runs/_cross_dataset_patch/attribution.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / ".pylibs"
if LIB.exists():
    sys.path.insert(0, str(LIB))

FEAT = ROOT / "data" / "features" / "phikon-v2" / "patch_multidataset"
OUT = ROOT / "runs" / "_cross_dataset_patch"


def img_meta(paths: list[str]) -> dict:
    from PIL import Image
    widths, heights, sizes, modes = [], [], [], Counter()
    for p in paths:
        try:
            with Image.open(p) as im:
                widths.append(im.width)
                heights.append(im.height)
                modes[im.mode] += 1
        except Exception:  # noqa: BLE001
            continue
        try:
            sizes.append(Path(p).stat().st_size)
        except Exception:  # noqa: BLE001
            pass
    def stat(a):
        a = np.array(a, dtype=float)
        return {"n": int(len(a)), "mean": round(float(a.mean()), 1) if len(a) else None,
                "std": round(float(a.std()), 1) if len(a) else None,
                "min": float(a.min()) if len(a) else None,
                "max": float(a.max()) if len(a) else None,
                "unique": int(len(np.unique(a)))} if len(a) else {}
    return {"width": stat(widths), "height": stat(heights),
            "file_bytes": stat(sizes), "modes": dict(modes)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-meta", type=int, default=1200, help="每类最多读取多少张的元数据")
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    res: dict = {}
    features = {}
    for npz in sorted(FEAT.glob("*.npz")):
        d = np.load(npz, allow_pickle=True)
        X, y, paths = d["features"], d["labels"].astype(int), d["paths"].astype(str)
        features[npz.stem] = (X, y, paths)

    # H3：类内 vs 类间的图像元数据差异
    res["metadata_by_dataset"] = {}
    meta_matrix = {}
    for name, (X, y, paths) in features.items():
        by_lab = defaultdict(list)
        for p, lab in zip(paths, y):
            by_lab[int(lab)].append(p)
        entry = {"n_pos": len(by_lab[1]), "n_neg": len(by_lab[0])}
        for lab, tag in ((0, "benign"), (1, "malignant")):
            sub = by_lab[lab][: args.max_meta]
            m = img_meta(sub)
            entry[tag] = m
            # 供 H3 检验用的元数据向量
            ws = [m["width"].get("mean") or 0]
            hs = [m["height"].get("mean") or 0]
            fs = [m["file_bytes"].get("mean") or 0]
            meta_matrix.setdefault(name, {})[lab] = (ws[0], hs[0], fs[0])
        res["metadata_by_dataset"][name] = entry

    # H3 检验：仅用 (width,height,filesize) 的均值做分类——注意这是「类级统计」，
    # 因此改为逐图读取元数据并训练分类器（样本级）。
    print("H3 检验：逐图元数据（宽/高/文件大小）能否区分类别")
    meta_auc = {}
    for name, (X, y, paths) in features.items():
        n = min(len(paths), 800)
        idx = np.random.default_rng(0).choice(len(paths), n, replace=False)
        rows, labs = [], []
        for i in idx:
            try:
                with __import__("PIL.Image", fromlist=["Image"]).open(paths[i]) as im:
                    w, h = im.width, im.height
            except Exception:  # noqa: BLE001
                continue
            try:
                b = Path(paths[i]).stat().st_size
            except Exception:  # noqa: BLE001
                b = 0
            rows.append([w, h, b, w / max(h, 1)])
            labs.append(int(y[i]))
        if len(set(labs)) < 2:
            continue
        M = np.array(rows, dtype=float)
        M = (M - M.mean(0)) / (M.std(0) + 1e-9)
        labs = np.array(labs)
        oof = np.zeros(len(labs))
        # 元数据只有 4 维、无分组意义，用普通 KFold 即可（分组在这里不适用）
        from sklearn.model_selection import KFold
        for tr, te in KFold(n_splits=4, shuffle=True, random_state=0).split(M):
            clf = LogisticRegression(max_iter=1000)
            clf.fit(M[tr], labs[tr])
            oof[te] = clf.predict_proba(M[te])[:, 1]
        a = float(roc_auc_score(labs, oof))
        meta_auc[name] = round(a, 4)
        print(f"  {name}: 元数据 AUC = {a:.4f}  (接近 0.5 表示无捷径)")
    res["h3_metadata_only_auc"] = meta_auc

    # H1/H2：数据集整体嵌入分布与类间可分性
    res["embedding_structure"] = {}
    centroids = {}
    for name, (X, y, paths) in features.items():
        c0, c1 = X[y == 0].mean(0), X[y == 1].mean(0)
        centroids[name] = {"benign": c0, "malignant": c1}
        between = float(np.linalg.norm(c0 - c1))
        within = float(0.5 * (np.linalg.norm(X[y == 0] - c0, axis=1).mean()
                              + np.linalg.norm(X[y == 1] - c1, axis=1).mean()))
        res["embedding_structure"][name] = {
            "class_centroid_distance": round(between, 3),
            "within_class_spread": round(within, 3),
            "separation_ratio": round(between / max(within, 1e-9), 3),
            "global_mean_norm": round(float(np.linalg.norm(X.mean(0))), 3),
        }

    # 跨数据集的「良性质心」与「恶性质心」对齐情况：
    # 若 A 的良性质心更接近 B 的恶性质心，说明类别语义/组织构成存在错位（H2 证据）
    names = list(centroids)
    if len(names) == 2:
        a, b = names
        def cos(u, v):
            return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-12))
        res["cross_centroid_alignment"] = {
            f"{a}.benign vs {b}.benign": round(cos(centroids[a]["benign"], centroids[b]["benign"]), 4),
            f"{a}.benign vs {b}.malignant": round(cos(centroids[a]["benign"], centroids[b]["malignant"]), 4),
            f"{a}.malignant vs {b}.malignant": round(cos(centroids[a]["malignant"], centroids[b]["malignant"]), 4),
            f"{a}.malignant vs {b}.benign": round(cos(centroids[a]["malignant"], centroids[b]["benign"]), 4),
        }
        d_bb = float(np.linalg.norm(centroids[a]["benign"] - centroids[b]["benign"]))
        d_bm = float(np.linalg.norm(centroids[a]["benign"] - centroids[b]["malignant"]))
        res["centroid_distance_check"] = {
            f"|{a}.benign - {b}.benign|": round(d_bb, 3),
            f"|{a}.benign - {b}.malignant|": round(d_bm, 3),
            "semantic_misalignment_suspected": bool(d_bm < d_bb),
        }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "attribution.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["# 跨数据集迁移失败的归因分析", "",
             "## H3 检验：数据集伪影（图像尺寸/文件大小捷径）", "",
             "| 数据集 | 仅元数据的 AUC | 判读 |", "| --- | --- | --- |"]
    for k, v in meta_auc.items():
        verdict = "无捷径（接近 0.5）" if abs(v - 0.5) < 0.15 else "**存在捷径风险**"
        lines.append(f"| {k} | {v:.4f} | {verdict} |")
    lines += ["", "## 嵌入结构（类间可分性）", "",
              "| 数据集 | 类质心距离 | 类内散度 | 分离比 |", "| --- | --- | --- | --- |"]
    for k, v in res["embedding_structure"].items():
        lines.append(f"| {k} | {v['class_centroid_distance']} | {v['within_class_spread']} | "
                     f"{v['separation_ratio']} |")
    if "cross_centroid_alignment" in res:
        lines += ["", "## H2 检验：两类语义是否错位", "",
                  "| 比较 | 余弦相似度 |", "| --- | --- |"]
        for k, v in res["cross_centroid_alignment"].items():
            lines.append(f"| {k} | {v} |")
        lines += ["", "余弦相似度越高越「同向」。若 `A.benign` 与 `B.malignant` 的相似度**高于**",
                  "`A.benign` 与 `B.benign`，则强烈提示两个数据集的类别语义不一致（H2）。"]
        ck = res.get("centroid_distance_check", {})
        lines += ["", f"- 语义错位可疑：**{ck.get('semantic_misalignment_suspected')}**"]
    lines += ["", "## 结论与写作提示", "",
              "- 若 H3 检验接近 0.5 且分离比在各自数据集内都很高，则**跨数据集失败更可能来自真实域偏移",
              "  或类别语义差异，而非数据集伪影**；",
              "- **必须**在论文中明确：两个数据集的类别定义不同（子宫内膜 NE/EH/EP/EA vs 子宫 tumour/non-tumour），",
              "  因此跨数据集 AUC 不能被解读为纯粹的域偏移度量；",
              "- 要把跨数据集迁移作为域偏移证据，需要**类别定义一致**的配对数据集——这正是活检数据",
              "  这一缺口必须在论文 Limitations 中如实写明。"]
    (OUT / "attribution.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({k: v for k, v in res.items() if k != "metadata_by_dataset"},
                     indent=2, ensure_ascii=False)[:2000])
    print(f"\n-> {OUT/'attribution.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
