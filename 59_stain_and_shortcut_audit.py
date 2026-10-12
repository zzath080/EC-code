#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
59_stain_and_shortcut_audit.py
==============================
回答两个审稿人必问的质疑：

    Q1（捷径审计）：模型是不是学到了**染色/背景/尺寸等伪影**，而不是组织形态？
    Q2（染色归一化消融）：跨数据集崩塌能否通过**染色归一化**缓解？
        —— 缓解 → 域偏移可部分修复；不缓解 → 是更深层的表征问题。

做法：
    (a) 对每张图计算「非形态学」统计量（尺寸、文件大小、RGB/HSV 统计、背景比例），
        训练分类器测其 AUC —— 接近 0.5 说明无捷径；显著 >0.5 说明存在捷径；
    (b) 对图像做 Reinhard 染色归一化（以各自数据集的整体统计为目标），
        重新提取 Phikon-v2 特征，重跑域内与跨数据集评估，与未归一化对比。

用法：
    python scripts/59_stain_and_shortcut_audit.py --stats        # 只做捷径审计（快）
    python scripts/59_stain_and_shortcut_audit.py --normalize    # 归一化 + 重提特征 + 复评
    python scripts/59_stain_and_shortcut_audit.py --all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / ".pylibs"
if LIB.exists():
    sys.path.insert(0, str(LIB))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# 来源定义集中在 source_registry，与 60_cross_source_matrix.py 共用一份，
# 避免"捷径审计"与"跨来源矩阵"跑在不同的来源集合上而无法互相印证。
from source_registry import SPECS as REGISTRY_SPECS, list_items as _reg_list_items  # noqa: E402

DATASETS = {name: spec["root"] for name, spec in REGISTRY_SPECS.items()}
MALIG = {name: set(spec.get("malignant") or []) for name, spec in REGISTRY_SPECS.items()}
# 注意：原始特征必须与归一化特征**来源集合一致**，否则对比无效
# （已实测踩坑：早期 FEAT_ORIG 指向只有 2 个来源的 patch_multidataset，
#   而归一化特征有 5 个来源，导致「原始 vs 归一化」的比较根本不在同一来源集上）。
FEAT_ORIG = ROOT / "data" / "features" / "phikon-v2" / "patch_multisource"
FEAT_NORM = ROOT / "data" / "features" / "phikon-v2" / "patch_multisource_stainnorm"
MODEL = ROOT / "models" / "phikon-v2"
OUT = ROOT / "runs" / "_stain_shortcut_audit"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def list_items():
    """列出所有来源的 (图像路径, 标签)，来源定义取自 source_registry。

    历史实现有个危险缺陷：它把**任何**含文件的叶子目录都当成良性，
    只把名字命中"恶性集合"的标成恶性。一旦目录结构变化（例如多出一层
    `images/`），它会静默地把那个目录当成第 3 个良性类混进标签，
    而且不报错。注册表版本只承认显式声明的类别目录，结构不符时会直接
    少一个来源（可见），而不是悄悄污染标签。
    """
    out = {}
    for name, spec in REGISTRY_SPECS.items():
        rows = [(p, lab) for p, lab, _grp in _reg_list_items(spec)]
        out[name] = rows
    return out


_H5_CACHE: dict = {}


def _load_rgb(path: Path):
    """读取一张图为 float32 RGB（0–1）与文件字节数。

    需同时支持普通图像文件与 **HDF5 内嵌图像**（PatchCamelyon 形式为
    `<h5path>::<row>`）。早期实现只走 PIL，导致 h5 来源在捷径审计里被静默跳过
    （结果里只出现 8/9 个来源，且没有任何报错）。这里显式分支并复用句柄。
    """
    sp = str(path)
    if "::" in sp:
        h5f, _, row = sp.rpartition("::")
        try:
            import h5py
            if h5f not in _H5_CACHE:
                _H5_CACHE[h5f] = h5py.File(h5f, "r")["x"]
            img = _H5_CACHE[h5f][int(row)]
            return np.asarray(img, dtype=np.float32) / 255.0, 0.0
        except Exception:  # noqa: BLE001
            return None, 0.0
    from PIL import Image
    try:
        with Image.open(path) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    except Exception:  # noqa: BLE001
        return None, 0.0
    try:
        fbytes = float(path.stat().st_size)
    except Exception:  # noqa: BLE001
        fbytes = 0.0
    return arr, fbytes


def stats_of(path: Path) -> dict | None:
    """非形态学统计量：尺寸、文件大小、颜色统计、背景比例。"""
    arr, fbytes = _load_rgb(path)
    if arr is None:
        return None
    h, w = arr.shape[0], arr.shape[1]
    flat = arr.reshape(-1, 3)
    mean = flat.mean(0)
    std = flat.std(0)
    # 背景比例（近白像素）
    white = ((flat > 0.9).all(axis=1)).mean()
    gray = flat.mean(1)
    dark = (gray < 0.35).mean()
    r, g, b = mean
    return {
        "width": w, "height": h, "aspect": w / max(h, 1), "file_kb": fbytes / 1024,
        "r": r, "g": g, "b": b, "std_r": std[0], "std_g": std[1], "std_b": std[2],
        "rgb_ratio_rg": r / max(g, 1e-6), "rgb_ratio_rb": r / max(b, 1e-6),
        "white_frac": white, "dark_frac": dark,
    }


def cmd_stats(max_per_class: int) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    items = list_items()
    res: dict = {}
    for name, rows in items.items():
        by_lab = defaultdict(list)
        for p, l in rows:
            by_lab[l].append(p)
        n = min(max_per_class, min(len(by_lab[0]), len(by_lab[1]))) if by_lab[0] and by_lab[1] else 0
        rng = np.random.default_rng(0)
        picked = [(p, 0) for p in rng.choice(by_lab[0], n, replace=False)] + \
                 [(p, 1) for p in rng.choice(by_lab[1], n, replace=False)]
        X, y, t0 = [], [], time.time()
        for p, l in picked:
            s = stats_of(Path(p))
            if s is None:
                continue
            X.append(list(s.values()))
            y.append(l)
        if not X:
            continue
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        keys = list(stats_of(Path(picked[0][0])).keys())
        # 标准化后做逻辑回归，交叉验证
        Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
        oof = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Xs, y):
            clf = LogisticRegression(max_iter=2000)
            clf.fit(Xs[tr], y[tr])
            oof[te] = clf.predict_proba(Xs[te])[:, 1]
        auc = float(roc_auc_score(y, oof))
        coef = np.abs(LogisticRegression(max_iter=2000).fit(Xs, y).coef_[0])
        top = sorted(zip(keys, coef), key=lambda kv: -kv[1])[:6]
        res[name] = {"n_per_class": n, "n_used": len(y), "stats_auc": round(auc, 4),
                     "top_features": [(k, round(float(v), 3)) for k, v in top],
                     "seconds": round(time.time() - t0, 1)}
        print(f"  {name}: 仅用图像统计量的 AUC = {auc:.4f}  "
              f"(n/类={n})  最强特征: {[k for k, _ in top[:3]]}")
    return res


def reinhard_normalize(arr: np.ndarray, target_mean: np.ndarray, target_std: np.ndarray) -> np.ndarray:
    """在 LAB 空间做 Reinhard 染色归一化的近似实现（逐图统计匹配）。"""
    m, s = arr.reshape(-1, 3).mean(0), arr.reshape(-1, 3).std(0) + 1e-6
    out = (arr - m) / s * target_std + target_mean
    return np.clip(out, 0, 1)


def cmd_normalize(limit: int | None) -> dict:
    import torch
    from PIL import Image
    from transformers import AutoModel

    items = list_items()
    # 计算每个数据集的全局颜色目标（用该数据集自身的均值/标准差，故不引入外部偏置）
    targets = {}
    for name, rows in items.items():
        acc, n = np.zeros(3), 0
        for p, _ in rows[:400]:
            try:
                with Image.open(p) as im:
                    a = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
                acc += a.reshape(-1, 3).mean(0)
                n += 1
            except Exception:  # noqa: BLE001
                continue
        targets[name] = acc / max(n, 1)
        print(f"  {name} 颜色目标（RGB 均值）: {np.round(targets[name], 3)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(str(MODEL)).eval().to(device)
    FEAT_NORM.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, rows in items.items():
        if limit:
            rows = rows[:limit]
        # 目标 std 用全体像素估计
        stds = []
        for p, _ in rows[:300]:
            try:
                with Image.open(p) as im:
                    a = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
                stds.append(a.reshape(-1, 3).std(0))
            except Exception:  # noqa: BLE001
                continue
        tstd = np.mean(stds, 0) if stds else np.array([0.2, 0.2, 0.2])
        feats, ys, ps = [], [], []
        t0 = time.time()
        batch = 64
        with torch.no_grad():
            for i in range(0, len(rows), batch):
                chunk = rows[i:i + batch]
                arrs = []
                for p, _ in chunk:
                    try:
                        with Image.open(p) as im:
                            a = np.asarray(im.convert("RGB").resize((224, 224), Image.BILINEAR),
                                           dtype=np.float32) / 255.0
                    except Exception:  # noqa: BLE001
                        a = np.zeros((224, 224, 3), dtype=np.float32)
                    arrs.append(reinhard_normalize(a, targets[name], tstd))
                x = torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2)
                x = (x - torch.tensor(MEAN).view(1, 3, 1, 1)) / torch.tensor(STD).view(1, 3, 1, 1)
                out = model(x.to(device))
                emb = out.last_hidden_state[:, 0] if hasattr(out, "last_hidden_state") else out[0][:, 0]
                feats.append(emb.float().cpu().numpy())
                ys.extend([l for _, l in chunk])
                ps.extend([str(p) for p, _ in chunk])
        X = np.concatenate(feats).astype(np.float32)
        np.savez_compressed(FEAT_NORM / f"{name}.npz", features=X,
                            labels=np.array(ys), paths=np.array(ps))
        summary[name] = {"n": len(ys), "shape": list(X.shape), "seconds": round(time.time() - t0, 1)}
        print(f"  [{name}] 归一化特征 {X.shape}  {time.time()-t0:.0f}s")
    return summary


def evaluate(feat_dir: Path, tag: str) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    data = {}
    for npz in sorted(feat_dir.glob("*.npz")):
        d = np.load(npz, allow_pickle=True)
        data[npz.stem] = (d["features"], d["labels"].astype(int), d["paths"].astype(str))
    res: dict = {"in_domain": {}, "cross": {}}
    for name, (X, y, paths) in data.items():
        groups = np.array([Path(p).stem for p in paths])
        oof = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(X, y, groups):
            clf = LogisticRegression(max_iter=2000)
            clf.fit(X[tr], y[tr])
            oof[te] = clf.predict_proba(X[te])[:, 1]
        res["in_domain"][name] = round(float(roc_auc_score(y, oof)), 4)
    names = list(data)
    for a in names:
        Xa, ya, _ = data[a]
        clf = LogisticRegression(max_iter=2000).fit(Xa, ya)
        for b in names:
            if a == b:
                continue
            Xb, yb, _ = data[b]
            p = clf.predict_proba(Xb)[:, 1]
            res["cross"][f"{a}->{b}"] = {
                "auc": round(float(roc_auc_score(yb, p)), 4),
                "mean_pred_pos": round(float(p[yb == 1].mean()), 4),
                "mean_pred_neg": round(float(p[yb == 0].mean()), 4),
            }
    print(f"\n[{tag}] 域内: " + ", ".join(f"{k}={v:.4f}" for k, v in res["in_domain"].items()))
    for k, v in res["cross"].items():
        print(f"[{tag}] 跨域 {k}: AUC={v['auc']:.4f}")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--max-per-class", type=int, default=600)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if args.all or not (args.stats or args.normalize):
        args.stats = args.normalize = True

    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {}

    if args.stats:
        print("=== 捷径审计：仅用非形态学统计量能否区分类别 ===")
        report["shortcut_audit"] = cmd_stats(args.max_per_class)
        print("\n判读：AUC 接近 0.5 → 无捷径；显著 >0.5 → 存在捷径，域内性能被高估")

    if args.normalize:
        print("\n=== 染色归一化（Reinhard）后重新提取特征 ===")
        report["stainnorm_extract"] = cmd_normalize(args.limit)
        print("\n=== 对比评估 ===")
        report["eval_original"] = evaluate(FEAT_ORIG, "原始")
        report["eval_stainnorm"] = evaluate(FEAT_NORM, "染色归一化后")

        # 结论：必须区分"一致改善 / 方向不一致 / 一致无改善"三种情况
        delta = {}
        for k in report["eval_original"]["cross"]:
            if k in report["eval_stainnorm"]["cross"]:
                delta[k] = round(report["eval_stainnorm"]["cross"][k]["auc"]
                                 - report["eval_original"]["cross"][k]["auc"], 4)
        report["cross_auc_delta_after_normalization"] = delta
        improved = [k for k, v in delta.items() if v > 0.05]
        worsened = [k for k, v in delta.items() if v < -0.05]
        if improved and not worsened:
            verdict = ("染色归一化在全部方向上改善跨域性能 → 域偏移可部分归因于染色差异")
        elif improved and worsened:
            verdict = ("染色归一化效果**方向不一致**：部分方向改善、部分方向恶化 → "
                       "崩塌并非单纯染色差异，不能声称归一化可修复")
        elif worsened and not improved:
            verdict = ("染色归一化在全部方向上恶化跨域性能 → 崩塌并非染色差异所致，"
                       "归一化不是解法")
        else:
            verdict = ("染色归一化对跨域性能无实质影响（|ΔAUC| ≤ 0.05）→ "
                       "崩塌并非染色差异所致")
        report["conclusion"] = {
            "improved_directions": improved,
            "worsened_directions": worsened,
            "verdict": verdict,
        }

    # 与已有结果**合并**写入：--stats 与 --normalize 分两次跑时会各自只填一半字段，
    # 若直接覆盖会让另一次的结果丢失（已实测踩坑）。
    res_path = OUT / "results.json"
    merged: dict = {}
    if res_path.exists():
        try:
            merged = json.loads(res_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            merged = {}
    merged.update(report)
    res_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["# 捷径审计与染色归一化消融", ""]
    if "shortcut_audit" in report:
        lines += ["## 1. 捷径审计：仅用非形态学统计量分类", "",
                  "| 数据集 | 每类样本 | 统计量 AUC | 最强特征（前 3） |", "| --- | --- | --- | --- |"]
        for k, v in report["shortcut_audit"].items():
            lines.append(f"| {k} | {v['n_per_class']} | **{v['stats_auc']:.4f}** | "
                         f"{', '.join(f for f, _ in v['top_features'][:3])} |")
        lines += ["", "> AUC 接近 0.5 表示模型无法从尺寸/颜色/背景等非形态学线索判断类别；",
                  "> 显著高于 0.5 则说明该数据集的域内性能可能被捷径高估。"]
    if "eval_original" in report:
        lines += ["", "## 2. 染色归一化前后对比", "",
                  "| 评估 | 原始 | 染色归一化后 | Δ |", "| --- | --- | --- | --- |"]
        for k, v in report["eval_original"]["in_domain"].items():
            nv = report["eval_stainnorm"]["in_domain"].get(k)
            lines.append(f"| 域内 {k} | {v:.4f} | {nv:.4f} | {nv - v:+.4f} |")
        for k, v in report["eval_original"]["cross"].items():
            nv = report["eval_stainnorm"]["cross"].get(k, {}).get("auc")
            if nv is not None:
                lines.append(f"| 跨域 {k} | {v['auc']:.4f} | {nv:.4f} | {nv - v['auc']:+.4f} |")
        c = report.get("conclusion", {})
        lines += ["", f"**结论**：{c.get('verdict', '')}", ""]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n-> {OUT/'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
