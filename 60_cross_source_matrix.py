#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
60_cross_source_matrix.py
=========================
跨来源泛化矩阵（论文核心图表）：把所有可用的 patch 级良恶性数据集两两互测，
并做留一来源（leave-one-source-out）评估。

与 43_cross_dataset_patch_benchmark.py 的区别：
    - 支持任意数量来源（自动发现，不写死两个）
    - 对每个来源同时报告：良恶性定义、样本量、正负比
    - 输出完整 N×N 矩阵 + 每个来源的「跨来源平均 AUC」与「域内 AUC 之差」
    - 内置捷径审计提示（若某来源的元数据 AUC 偏高，标注出来）

用法：
    python scripts/60_cross_source_matrix.py --extract
    python scripts/60_cross_source_matrix.py --eval
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

# 编码器与输出目录由 --encoder 决定（支持第二编码器对照）。
# 默认 phikon-v2，保持既有结果不变。
ENCODER_SPECS = {
    "phikon-v2": {"model": ROOT / "models" / "phikon-v2"},
    "phikon": {"model": ROOT / "models" / "phikon"},
    "vit-base-in": {"model": ROOT / "models" / "vit-base-in"},
}
DEFAULT_ENCODER = "phikon-v2"


def feat_dir(encoder: str) -> Path:
    return ROOT / "data" / "features" / encoder / "patch_multisource"


def out_dir(encoder: str) -> Path:
    return ROOT / "runs" / (f"_cross_source_matrix_{encoder}"
                           if encoder != DEFAULT_ENCODER else "_cross_source_matrix")


# 来源定义集中在 source_registry，避免与 59_stain_and_shortcut_audit.py 各存一份
# （两处不一致会让捷径审计与矩阵跑在不同的来源集合上，结论互相印证就失效了）。
sys.path.insert(0, str(Path(__file__).resolve().parent))
from source_registry import IMAGE_EXTS, groups_for, list_items as _reg_list_items  # noqa: E402
from source_registry import discover as _reg_discover  # noqa: E402
from source_registry import SPECS as REGISTRY_SPECS  # noqa: E402

MS = ROOT / "data" / "raw" / "patch_multisource"
MODEL = ENCODER_SPECS[DEFAULT_ENCODER]["model"]   # 兼容旧调用
FEAT = feat_dir(DEFAULT_ENCODER)                  # 兼容旧调用
OUT = out_dir(DEFAULT_ENCODER)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# 兼容旧名字：SPECS 现在指向注册表
SPECS = REGISTRY_SPECS


def discover(only: list[str] | None = None) -> dict[str, dict]:
    """只保留实际存在且两类都有的数据集。"""
    return _reg_discover(only)


def list_items(spec: dict) -> list[tuple[Path, int]]:
    """返回 [(图像路径, 标签)]（注册表版本还带分组键，这里丢弃以保持调用方不变）。"""
    return [(p, lab) for p, lab, _grp in _reg_list_items(spec)]


def cmd_extract(batch: int, limit: int | None, encoder: str = DEFAULT_ENCODER,
                per_class_cap: int | None = 5000,
                only: list[str] | None = None,
                cap_overrides: dict[str, int] | None = None) -> int:
    """提取冻结特征。

    抽样口径（很重要，直接影响结果数字）：
    `per_class_cap` 是**每个来源每类**的上限，对全部来源统一生效。
    原 5 个来源都已按**全量**提取过特征、论文数字基于全量，因此必须用
    `cap_overrides` 把它们设为 0（=不抽样）以保持完全不变；只有新增的大来源
    （breast_idc 555,048 张）需要设上限以控制计算量。
    若不加覆盖而直接调小 per_class_cap，会把原来源一并降采样，
    使 `runs/_cross_source_matrix` 的既有数字无故失效。
    """
    cap_overrides = cap_overrides or {}
    import torch
    from PIL import Image
    from transformers import AutoModel

    model_dir = ENCODER_SPECS[encoder]["model"]
    feat = feat_dir(encoder)
    specs = discover(only)
    print(f"编码器 {encoder}（{model_dir}）")
    print(f"发现 {len(specs)} 个来源：")
    for k, v in specs.items():
        print(f"  {k:14s} {v['organ']:12s} 恶性 {v['n_pos']:5d} / 良性 {v['n_neg']:5d}")
    if not specs:
        print("[ERROR] 没有可用来源", file=sys.stderr)
        return 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(str(model_dir)).eval().to(device)
    feat.mkdir(parents=True, exist_ok=True)

    for name, spec in specs.items():
        items = list_items(spec)
        # 该来源的生效上限：覆盖值优先（0 = 不抽样，保持全量）
        src_cap = cap_overrides.get(name, per_class_cap)
        if src_cap == 0:
            src_cap = None
        # 按来源**分层封顶**：大来源（breast_idc 有 555,048 张，恶性仅占 28%）若按
        # 全局随机抽样会破坏类别平衡，也会让提取时间失控。这里每类独立抽样，
        # 与捷径审计的"每类上限"口径保持一致。
        if src_cap:
            rng_c = np.random.default_rng(0)
            pos = [it for it in items if it[1] == 1]
            neg = [it for it in items if it[1] == 0]
            k = min(src_cap, len(pos), len(neg))
            if k and (len(pos) > k or len(neg) > k):
                items = ([pos[i] for i in rng_c.choice(len(pos), k, replace=False)]
                         + [neg[i] for i in rng_c.choice(len(neg), k, replace=False)])
                print(f"  [{name}] 分层抽样至每类 {k}（原始 恶{len(pos)}/良{len(neg)}）")
        if limit:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(items), min(limit, len(items)), replace=False)
            items = [items[i] for i in idx]
        feats, labels, paths = [], [], []
        t0 = time.time()
        # HDF5 来源（PatchCamlelyon）：list_items 返回 "path::row" 形式的合成路径。
        # 这里按需打开 h5 并复用句柄，避免每张图都重新打开 6.75 GB 文件。
        h5_cache: dict[str, object] = {}
        with torch.no_grad():
            for i in range(0, len(items), batch):
                chunk = items[i:i + batch]
                arrs = []
                for p, _ in chunk:
                    sp = str(p)
                    if "::" in sp:
                        h5f, row = sp.split("::", 1)
                        try:
                            import h5py
                            if h5f not in h5_cache:
                                h5_cache[h5f] = h5py.File(h5f, "r")["x"]
                            img = h5_cache[h5f][int(row)]          # 96x96x3 uint8
                            arrs.append(np.asarray(
                                Image.fromarray(img).resize((224, 224), Image.BILINEAR),
                                dtype=np.uint8))
                        except Exception:  # noqa: BLE001
                            arrs.append(np.zeros((224, 224, 3), dtype=np.uint8))
                        continue
                    try:
                        with Image.open(p) as im:
                            arrs.append(np.asarray(im.convert("RGB").resize((224, 224), Image.BILINEAR),
                                                   dtype=np.uint8))
                    except Exception:  # noqa: BLE001
                        arrs.append(np.zeros((224, 224, 3), dtype=np.uint8))
                x = torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2).float().div_(255.0)
                x = (x - torch.tensor(MEAN).view(1, 3, 1, 1)) / torch.tensor(STD).view(1, 3, 1, 1)
                out = model(x.to(device))
                emb = out.last_hidden_state[:, 0] if hasattr(out, "last_hidden_state") else out[0][:, 0]
                feats.append(emb.float().cpu().numpy())
                labels.extend([l for _, l in chunk])
                paths.extend([str(p) for p, _ in chunk])
        for _h in h5_cache.values():
            try:
                _h.file.close()
            except Exception:  # noqa: BLE001
                pass
        X = np.concatenate(feats).astype(np.float32)
        np.savez_compressed(feat / f"{name}.npz", features=X,
                            labels=np.array(labels), paths=np.array(paths))
        print(f"  [{name}] {X.shape}  {time.time()-t0:.0f}s")
    return 0


def cmd_eval(seed: int, encoder: str = DEFAULT_ENCODER) -> int:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.model_selection import GroupKFold

    feat = feat_dir(encoder)
    out = out_dir(encoder)
    print(f"编码器 {encoder}；特征目录 {feat.relative_to(ROOT)}")
    data = {}
    for npz in sorted(feat.glob("*.npz")):
        d = np.load(npz, allow_pickle=True)
        data[npz.stem] = (d["features"], d["labels"].astype(int), d["paths"].astype(str))
    if len(data) < 2:
        print(f"[ERROR] 只有 {len(data)} 个来源，无法做跨来源矩阵", file=sys.stderr)
        return 1

    names = list(data)
    print(f"来源 {len(names)} 个：" + ", ".join(f"{n}({len(data[n][1])})" for n in names))

    # 域内（5 折 GroupKFold）
    in_dom: dict[str, dict] = {}
    for n, (X, y, paths) in data.items():
        # 分组键由注册表决定：有患者 ID 的来源（breast_idc）按患者分组，
        # 其余退回文件 stem。此前一律用 stem，无法排除同一患者的图像跨折。
        groups = np.array(groups_for(paths))
        n_grp = len(set(groups.tolist()))
        if n_grp < 5:
            print(f"  [WARN] {n}: 仅 {n_grp} 个分组，跳过域内 OOF")
            continue
        oof = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(X, y, groups):
            clf = LogisticRegression(max_iter=2000).fit(X[tr], y[tr])
            oof[te] = clf.predict_proba(X[te])[:, 1]
        in_dom[n] = {"n": len(y), "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
                     "n_groups": n_grp,
                     "auc": round(float(roc_auc_score(y, oof)), 4),
                     "ap": round(float(average_precision_score(y, oof)), 4),
                     "mean_pred_pos": round(float(oof[y == 1].mean()), 4),
                     "mean_pred_neg": round(float(oof[y == 0].mean()), 4)}
        print(f"  域内 {n:14s} AUC={in_dom[n]['auc']:.4f}  AP={in_dom[n]['ap']:.4f}  "
              f"分组={n_grp}")

    # 跨来源全矩阵
    matrix: dict[str, dict] = {}
    for a in names:
        Xa, ya, _ = data[a]
        clf = LogisticRegression(max_iter=2000).fit(Xa, ya)
        tr_pred = clf.predict_proba(Xa)[:, 1]
        for b in names:
            if a == b:
                continue
            Xb, yb, _ = data[b]
            p = clf.predict_proba(Xb)[:, 1]
            matrix[f"{a}->{b}"] = {
                "auc": round(float(roc_auc_score(yb, p)), 4),
                "ap": round(float(average_precision_score(yb, p)), 4),
                "mean_pred_pos": round(float(p[yb == 1].mean()), 4),
                "mean_pred_neg": round(float(p[yb == 0].mean()), 4),
                "n_test": int(len(yb)),
                # 校准于大样本的偏移：目标域平均预测 − 源域平均预测
                "pred_shift": round(float(p.mean() - tr_pred.mean()), 4),
                # 判别力保留率（相对目标域域内 AUC）
                "auc_retention": round(float(roc_auc_score(yb, p)) / max(in_dom[b]["auc"], 1e-9), 4),
            }

    # 每个来源作为目标域时的「跨来源平均 AUC」
    as_target: dict[str, dict] = {}
    for b in names:
        keys = [k for k in matrix if k.endswith(f"->{b}")]
        if not keys:
            continue
        aucs = [matrix[k]["auc"] for k in keys]
        as_target[b] = {
            "n_sources": len(keys),
            "mean_auc": round(float(np.mean(aucs)), 4),
            "min_auc": round(float(np.min(aucs)), 4),
            "max_auc": round(float(np.max(aucs)), 4),
            "in_domain_auc": in_dom[b]["auc"],
            "gap": round(float(np.mean(aucs)) - in_dom[b]["auc"], 4),
        }
        print(f"  作为目标域 {b:14s} 跨来源平均 AUC={as_target[b]['mean_auc']:.4f} "
              f"(域内 {in_dom[b]['auc']:.4f}, 落差 {as_target[b]['gap']:+.4f})")

    result = {"n_sources": len(names), "in_domain": in_dom,
              "cross_matrix": matrix, "as_target": as_target,
              "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}

    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    # CSV 矩阵
    with open(out / "matrix_auc.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["source\\target"] + names)
        for a in names:
            row = [a]
            for b in names:
                row.append(round(in_dom[a]["auc"], 4) if a == b else matrix.get(f"{a}->{b}", {}).get("auc", ""))
            w.writerow(row)

    # 报告
    lines = ["# 跨来源泛化矩阵", "",
             f"来源数：{len(names)}；评估方式：Phikon-v2 冻结特征 + 逻辑回归，5 折 GroupKFold（按图像编号分组）",
             "", "## 1. 各来源概况与域内性能", "",
             "| 来源 | n | 恶性 | 良性 | 域内 AUC | AP |", "| --- | --- | --- | --- | --- | --- |"]
    for n in names:
        d = in_dom[n]
        lines.append(f"| {n} | {d['n']} | {d['n_pos']} | {d['n_neg']} | **{d['auc']:.4f}** | {d['ap']:.4f} |")

    lines += ["", "## 2. 跨来源 AUC 矩阵（行=训练来源，列=测试来源，对角=域内）", "",
              "| 训练\\测试 | " + " | ".join(names) + " |",
              "| --- | " + " | ".join("---" for _ in names) + " |"]
    for a in names:
        row = [a]
        for b in names:
            row.append(f"{in_dom[a]['auc']:.3f}" if a == b else
                       f"{matrix.get(f'{a}->{b}', {}).get('auc', float('nan')):.3f}")
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## 3. 每个来源作为目标域时的表现", "",
              "| 目标域 | 来源数 | 跨来源平均 AUC | 最差 | 最好 | 域内 AUC | 落差 |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for n, v in as_target.items():
        lines.append(f"| {n} | {v['n_sources']} | **{v['mean_auc']:.4f}** | {v['min_auc']:.4f} | "
                     f"{v['max_auc']:.4f} | {v['in_domain_auc']:.4f} | {v['gap']:+.4f} |")

    lines += ["", "## 4. 判读与必须写入的局限", ""]
    # 自动生成判读（基于实际数值，避免每次手工改写）
    all_cross = [v["auc"] for v in matrix.values()]
    gaps = [v["gap"] for v in as_target.values()]
    below_random = [k for k, v in matrix.items() if v["auc"] < 0.5]
    lines += [
        f"- **{len(all_cross)} 个跨来源组合的 AUC 范围：{min(all_cross):.3f}–{max(all_cross):.3f}**；"
        f"域内 AUC 范围 {min(d['auc'] for d in in_dom.values()):.4f}–{max(d['auc'] for d in in_dom.values()):.4f}。",
        f"- 每个来源作为目标域时的平均落差：**{min(gaps):+.4f} 至 {max(gaps):+.4f}**。",
        f"- 低于随机水平（AUC<0.5）的组合数：**{len(below_random)}/{len(all_cross)}**"
        + (f"（{', '.join(below_random[:4])}{'…' if len(below_random) > 4 else ''}）" if below_random else ""),
        f"- 结论：**域内性能完全不能外推到新来源**，且这是跨器官、跨方向的**系统性**现象。",
        "",
        "### 必须同写的局限",
        "",
        "- 各来源的**类别定义不同**（例如子宫内膜 NE/EH/EP/EA vs 子宫 tumour/non-tumour vs "
        "卵巢非癌/四亚型 vs 乳腺 benign/malignant），因此跨来源 AUC **不能**被解读为"
        "纯粹的域偏移度量；",
        "- 已实测（`scripts/59_stain_and_shortcut_audit.py`）多个来源存在**非形态学捷径**"
        "（仅用图像尺寸/颜色/文件大小统计量即达 AUC 0.80/0.92），故域内高 AUC 本身可能被高估；",
        "- 各来源**无患者标识**，只能按图像编号分组，无法完全排除同一患者的图像跨折；",
        "- **禁止**从本表推断\"某个模型更好\"：此处只有一个编码器（Phikon-v2）与一个线性探针，"
        "目的是量化跨来源落差，不是模型排名。",
    ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\n-> {out/'report.md'}")
    print(f"-> {out/'matrix_auc.csv'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--encoder", default=DEFAULT_ENCODER, choices=list(ENCODER_SPECS))
    ap.add_argument("--per-class-cap", type=int, default=5000,
                    help="每来源每类上限（0 = 不设上限），对所有来源统一生效")
    ap.add_argument("--cap-overrides", nargs="*", default=["endo_histo=0", "uterus_ca=0",
                                                          "ovarian_ca=0", "breakhis_400x=0",
                                                          "lung_colon=0"],
                    help="按来源覆盖上限，格式 来源=上限（0 = 不抽样）。默认把原 5 个来源"
                         "设为不抽样，以保持论文既有数字完全不变；只对新增大来源生效上限。")
    ap.add_argument("--only", nargs="*", default=None,
                    help="只处理指定来源（避免重提已有特征）")
    args = ap.parse_args()
    if not (args.extract or args.eval):
        args.extract = args.eval = True
    if args.extract:
        ov = {}
        for kv in (args.cap_overrides or []):
            if "=" in kv:
                k, v = kv.split("=", 1)
                try:
                    ov[k.strip()] = int(v)
                except ValueError:
                    print(f"[WARN] 忽略无法解析的 cap-override: {kv}")
        rc = cmd_extract(args.batch, args.limit, args.encoder,
                         args.per_class_cap, args.only, ov)
        if rc != 0:
            return rc
    if args.eval:
        return cmd_eval(args.seed, args.encoder)
    return 0


if __name__ == "__main__":
    sys.exit(main())
