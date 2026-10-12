"""捷径残差化：量化"域内 AUC 里有多少来自非形态学线索"。

动机
----
`Results` §3.4 证明了两件事：① 7/7 来源的标签能被**纯图像统计量**预测（捷径 AUC 0.724–0.978）；
② 域内 AUC 与捷径 AUC 秩相关 ρ=+0.857。

但论文**始终没有量化"高估了多少"**——只说"部分来自捷径"。这留下一个可被审稿人直接攻击的缺口：
既然你主张捷径抬高了域内 AUC，那就把捷径成分去掉，看域内 AUC 掉多少。

做法（下界，非因果）
------------------
对每个来源：
  1. 用该来源**训练折**的图像统计量拟合岭回归，预测**特征向量的每一个维度**
     （即"特征中可由非形态学线索线性解释的部分"）；
  2. 在**验证折**上做残差化：X_res = X − X_hat；
  3. 在残差特征上重跑同一套逻辑回归探针与同一套 5 折 GroupKFold；
  4. 与原始域内 AUC 比较。

为什么是"下界"：线性残差化只能移除**线性可解释**的捷径成分；非线性可解释部分仍然保留。
因此 ΔAUC 是"捷径贡献"的**下界估计**，不是完整归因。

刻意不做的事：不声称因果。残差化同时会移除与捷径**共线但真实**的形态学信号，
因此本实验给出的是"在移除线性伪影可解释成分后还剩多少排序能力"，仅此而已。

用法：
    python scripts/91_shortcut_residualization.py
输出：
    runs/_shortcut_residual/baseline.json
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / ".pylibs"
if LIB.exists():
    sys.path.insert(0, str(LIB))
sys.path.insert(0, str(ROOT / "scripts"))

from source_registry import SPECS, group_key_for, list_items  # noqa: E402

RUNS = ROOT / "runs"
OUT = RUNS / "_shortcut_residual"
FEAT = ROOT / "data" / "features" / "phikon-v2" / "patch_multisource"
MAX_PER_CLASS = 600          # 与捷径审计同口径，保证可比
SEED = 0


# 复用 59 的 stats_of，而不是在本脚本里再复制一份。
# 复制的那一份没有 h5 支持，导致 PatchCamelyon 的 `<h5path>::<row>` 全部读不出、
# X 变成空数组，随后 GroupKFold 抛 ValueError 又被过宽的 except 误报成"分组数不足"。
# 统计量必须与 59 完全一致，否则"残差化掉的成分"与"审计测到的捷径"不是同一个量。
from importlib import import_module  # noqa: E402
_s59 = import_module("59_stain_and_shortcut_audit")


def stats_of(path: Path) -> list[float] | None:
    """非形态学图像统计量（与 59 的捷径审计**同一定义**，保证可比）。"""
    d = _s59.stats_of(Path(path))
    if d is None:
        return None
    keys = ["width", "height", "aspect", "file_kb", "r", "g", "b",
            "std_r", "std_g", "std_b", "rgb_ratio_rg", "rgb_ratio_rb",
            "white_frac", "dark_frac"]
    return [float(d[k]) for k in keys]


def main() -> int:
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    OUT.mkdir(parents=True, exist_ok=True)
    res: dict = {"config": {"max_per_class": MAX_PER_CLASS, "seed": SEED,
                            "encoder": "phikon-v2", "ridge_alpha": 1.0},
                 "note": ("下界估计：线性残差化只移除线性可解释的捷径成分。"
                          "不声称因果——残差化同时会移除与捷径共线但真实的形态学信号。"),
                 "sources": {}}

    for name, spec in SPECS.items():
        npz = FEAT / f"{name}.npz"
        if not npz.exists():
            print(f"[SKIP] {name}: 缺 {npz.name}")
            continue
        d = np.load(npz, allow_pickle=True)
        X_all, y_all = d["features"].astype(np.float32), d["labels"].astype(int)
        paths = d["paths"].astype(str)

        # 与捷径审计同口径的分层抽样
        rng = np.random.default_rng(SEED)
        idx_by = defaultdict(list)
        for i, (yy, pp) in enumerate(zip(y_all, paths)):
            idx_by[int(yy)].append(i)
        k = min(MAX_PER_CLASS, len(idx_by[0]), len(idx_by[1]))
        pick = ([idx_by[0][i] for i in rng.choice(len(idx_by[0]), k, replace=False)]
                + [idx_by[1][i] for i in rng.choice(len(idx_by[1]), k, replace=False)])
        pick = sorted(pick)
        X, y = X_all[pick], y_all[pick]
        ps = [paths[i] for i in pick]

        # 图像统计量
        S, keep = [], []
        t0 = time.time()
        for j, p in enumerate(ps):
            s = stats_of(Path(p))
            if s is not None:
                S.append(s)
                keep.append(j)
        S = np.asarray(S, dtype=np.float64)
        X, y = X[keep], y[keep]
        ps = [ps[j] for j in keep]
        S = (S - S.mean(0)) / (S.std(0) + 1e-9)      # 标准化，岭回归更稳

        # 分组键必须走 group_key_for：h5 来源（PCam）的路径形如 `*.h5::<row>`，
        # 直接取 Path(p).stem 只会得到行号，导致分组数退化成"每张图一组"以外的错误层级
        # （实测表现为 "[SKIP] 分组数不足"）。
        groups = np.array([group_key_for(p) for p in ps])
        # 区分两种失败：样本/分组真的不足，与别处异常。过宽的 except 会把
        # "X 为空"之类的问题伪装成"分组数不足"，浪费排查时间（已踩）。
        if len(X) == 0 or len(set(y.tolist())) < 2:
            print(f"[SKIP] {name}: 可用样本 {len(X)}，类别 {sorted(set(y.tolist()))}")
            continue
        if len(set(groups.tolist())) < 5:
            print(f"[SKIP] {name}: 分组仅 {len(set(groups.tolist()))} 个")
            continue
        folds = list(GroupKFold(5).split(X, y, groups))

        auc_orig, auc_res, auc_shortcut = [], [], []
        for tr, te in folds:
            # 探针：原始特征 vs 残差特征，折划分完全相同
            clf = LogisticRegression(max_iter=2000).fit(X[tr], y[tr])
            auc_orig.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))

            # 岭回归：用统计量预测特征的每一维（仅在训练折上拟合）
            ridge = Ridge(alpha=1.0).fit(S[tr], X[tr])
            X_res = X - ridge.predict(S)             # 全样本残差化（线性投影）
            clf2 = LogisticRegression(max_iter=2000).fit(X_res[tr], y[tr])
            auc_res.append(roc_auc_score(y[te], clf2.predict_proba(X_res[te])[:, 1]))

            # 仅用统计量本身的探针（对照，应与捷径审计一致）
            clf3 = LogisticRegression(max_iter=2000).fit(S[tr], y[tr])
            auc_shortcut.append(roc_auc_score(y[te], clf3.predict_proba(S[te])[:, 1]))

        o, r, sc = float(np.mean(auc_orig)), float(np.mean(auc_res)), float(np.mean(auc_shortcut))
        res["sources"][name] = {
            "n": int(len(y)), "n_per_class": int(k),
            "auc_original": round(o, 4),
            "auc_after_residualization": round(r, 4),
            "delta_auc": round(r - o, 4),
            "auc_statistics_only": round(sc, 4),
            "seconds": round(time.time() - t0, 1),
        }
        print(f"  {name:18s} 原始 {o:.4f} -> 残差化 {r:.4f}  (Δ {r-o:+.4f})   "
              f"仅统计量 {sc:.4f}")

    vals = [v["delta_auc"] for v in res["sources"].values()]
    if vals:
        res["summary"] = {
            "n_sources": len(vals),
            "mean_delta_auc": round(float(np.mean(vals)), 4),
            "min_delta_auc": round(float(np.min(vals)), 4),
            "max_delta_auc": round(float(np.max(vals)), 4),
            "all_negative": bool(all(v < 0 for v in vals)),
        }
        print(f"\n平均 ΔAUC = {res['summary']['mean_delta_auc']:+.4f}"
              f"（范围 {res['summary']['min_delta_auc']:+.4f} ~ "
              f"{res['summary']['max_delta_auc']:+.4f}）")
        print("判读：ΔAUC 为负说明移除线性可解释的非形态学成分后域内排序能力下降；"
              "这是捷径贡献的**下界**，不是因果归因。")

    (OUT / "baseline.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "baseline.json").read_text(encoding="utf-8")   # 写后校验可解码
    print(f"\n-> {OUT/'baseline.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
