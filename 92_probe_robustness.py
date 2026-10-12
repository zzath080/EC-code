"""探针稳健性：跨来源崩塌是否为"默认线性探针"的产物？

为什么必须做
------------
`Results` §3.3–3.5 的全部结论（42 配对崩塌、捷径审计、染色消融复评）
都基于**同一个** `sklearn.LogisticRegression(max_iter=2000)`——scikit-learn 全默认超参。
这让一个直接的质疑无法回答："你怎么知道崩塌不是这个弱线性探针造成的？"
本脚本正面回答：固定特征与折划分，只换探针，看结论是否改变。

设计原则（保证可比）
-------------------
1. **特征、标签、抽样、折划分完全不变**：复用 `runs/_cross_source_matrix/results.json`
   所用的同一批 npz 特征；域内仍是 5 折 GroupKFold（患者/文件名分组，与主结果一致）。
2. **测试集始终是全部目标样本**，不做子采样——AUC 因此在探针之间严格可比。
3. **训练集在计算不可行时做分层子采样**，并**显式记录实际训练量**：
   RBF-SVM 与 kNN 需要保留全部训练样本的核矩阵/邻域，25,000 样本的训练在单卡机器上不可行，
   故设训练上限；逻辑回归与 MLP 用全量。报告时同时给出训练量，避免"偷偷换数据"。
4. MLP 使用**固定随机种子**并在训练前标准化特征（test 用 train 的均值/方差），
   这是 MLP/SVM 的必要预处理；逻辑回归不需要但同样受益，故对非树模型统一处理。

输出：
    runs/_probe_robustness/results.json
    runs/_probe_robustness/report.md

用法：
    python scripts/92_probe_robustness.py --probes lr mlp svm knn
    python scripts/92_probe_robustness.py --probes lr --limit-sources 2   # 冒烟测试
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / ".pylibs"
if LIB.exists():
    sys.path.insert(0, str(LIB))
sys.path.insert(0, str(ROOT / "scripts"))

from source_registry import group_key_for  # noqa: E402

FEAT = ROOT / "data" / "features" / "phikon-v2" / "patch_multisource"
OUT = ROOT / "runs" / "_probe_robustness"

SEED = 0
# 训练集上限：仅对需要保留训练样本的探针生效（SVM 核矩阵、kNN 邻域）
TRAIN_CAP = {"lr": None, "mlp": None, "svm": 6000, "knn": 6000}


def make_probe(kind: str, seed: int = SEED):
    """返回 (fit_predict 可调用, 训练量说明)。"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC

    if kind == "lr":
        def f(Xtr, ytr, Xte):
            m = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
            return m.predict_proba(Xte)[:, 1]
        return f, "LogisticRegression(默认超参)"

    if kind == "mlp":
        def f(Xtr, ytr, Xte):
            m = MLPClassifier(hidden_layer_sizes=(128,), max_iter=400,
                              early_stopping=True, n_iter_no_change=20,
                              random_state=seed).fit(Xtr, ytr)
            return m.predict_proba(Xte)[:, 1]
        return f, "MLP(128, early stopping)"

    if kind == "svm":
        def f(Xtr, ytr, Xte):
            m = SVC(C=1.0, kernel="rbf", gamma="scale",
                    probability=False, random_state=seed).fit(Xtr, ytr)
            # 用 decision_function 排序即可算 AUC，避免 Platt 缩放的开销
            return m.decision_function(Xte)
        return f, "SVC(rbf, C=1, gamma=scale)"

    if kind == "knn":
        def f(Xtr, ytr, Xte):
            m = KNeighborsClassifier(n_neighbors=15, n_jobs=-1).fit(Xtr, ytr)
            return m.predict_proba(Xte)[:, 1]
        return f, "KNN(k=15)"

    raise ValueError(f"未知探针 {kind}")


def stratified_cap(idx: np.ndarray, y: np.ndarray, cap: int | None, rng) -> np.ndarray:
    """对训练索引做分层子采样，保持两类比例。"""
    if cap is None or len(idx) <= cap:
        return idx
    pos = idx[y[idx] == 1]
    neg = idx[y[idx] == 0]
    k_pos = max(1, int(round(cap * len(pos) / len(idx))))
    k_neg = max(1, cap - k_pos)
    sel = np.concatenate([
        rng.choice(pos, min(k_pos, len(pos)), replace=False),
        rng.choice(neg, min(k_neg, len(neg)), replace=False)])
    return np.sort(sel)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", nargs="+", default=["lr", "mlp", "svm", "knn"],
                    choices=["lr", "mlp", "svm", "knn"])
    ap.add_argument("--encoder", default="phikon-v2")
    ap.add_argument("--limit-sources", type=int, default=None,
                    help="只跑前 N 个来源（冒烟测试用）")
    args = ap.parse_args()

    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    feat_root = ROOT / "data" / "features" / args.encoder / "patch_multisource"
    if not feat_root.exists():
        print(f"[ERROR] 缺 {feat_root}", file=sys.stderr)
        return 2

    data: dict[str, tuple[np.ndarray, np.ndarray, list[str]]] = {}
    for npz in sorted(feat_root.glob("*.npz")):
        d = np.load(npz, allow_pickle=True)
        data[npz.stem] = (d["features"].astype(np.float32),
                          d["labels"].astype(int),
                          [str(p) for p in d["paths"]])
    names = sorted(data)
    if args.limit_sources:
        names = names[: args.limit_sources]
    if len(names) < 2:
        print("[ERROR] 少于 2 个来源", file=sys.stderr)
        return 2

    print(f"来源 {len(names)} 个：" + ", ".join(f"{n}({len(data[n][1])})" for n in names))
    print(f"探针：{', '.join(args.probes)}\n")

    report: dict = {"encoder": args.encoder, "seed": SEED, "train_cap": TRAIN_CAP,
                    "probes": {}, "note": (
                        "测试集始终为全部目标样本，AUC 在探针间严格可比；"
                        "训练集在 TRAIN_CAP 限制下做分层子采样，实际训练量逐项记录。")}

    for kind in args.probes:
        probe, desc = make_probe(kind)
        rng = np.random.default_rng(SEED)
        cap = TRAIN_CAP[kind]
        print(f"=== 探针 {kind}: {desc}（训练上限 {cap or '全量'}）===")
        t_all = time.time()

        entry: dict = {"probe": kind, "description": desc, "train_cap": cap,
                       "in_domain": {}, "cross": {}, "n_train_used": {}}

        # ---- 域内：5 折 GroupKFold
        for n in names:
            X, y, ps = data[n]
            groups = np.array([group_key_for(p) for p in ps])
            if len(set(groups.tolist())) < 5:
                print(f"  [WARN] {n}: 分组不足，跳过域内")
                continue
            oof = np.zeros(len(y))
            tr_used = []
            for tr, te in GroupKFold(5).split(X, y, groups):
                tr2 = stratified_cap(tr, y, cap, rng)
                tr_used.append(len(tr2))
                mu, sd = X[tr2].mean(0), X[tr2].std(0) + 1e-6
                oof[te] = probe((X[tr2] - mu) / sd, y[tr2], (X[te] - mu) / sd)
            entry["in_domain"][n] = round(float(roc_auc_score(y, oof)), 4)
            entry["n_train_used"][n] = int(np.mean(tr_used))
            print(f"  域内 {n:18s} AUC={entry['in_domain'][n]:.4f}  "
                  f"训练量/折≈{entry['n_train_used'][n]}")

        # ---- 跨来源：全量训练，全量测试
        for a in names:
            Xa, ya, _ = data[a]
            ia = stratified_cap(np.arange(len(ya)), ya, cap, rng)
            mu, sd = Xa[ia].mean(0), Xa[ia].std(0) + 1e-6
            for b in names:
                if a == b:
                    continue
                Xb, yb, _ = data[b]
                if len(set(yb.tolist())) < 2:
                    continue
                p = probe((Xa[ia] - mu) / sd, ya[ia], (Xb - mu) / sd)
                entry["cross"][f"{a}->{b}"] = round(float(roc_auc_score(yb, p)), 4)

        ind = list(entry["in_domain"].values())
        crs = list(entry["cross"].values())
        entry["summary"] = {
            "n_sources": len(ind),
            "n_cross_pairs": len(crs),
            "in_domain_mean_auc": round(float(np.mean(ind)), 4),
            "in_domain_min_auc": round(float(np.min(ind)), 4),
            "cross_mean_auc": round(float(np.mean(crs)), 4),
            "cross_min_auc": round(float(np.min(crs)), 4),
            "cross_max_auc": round(float(np.max(crs)), 4),
            "n_below_chance": int(sum(1 for v in crs if v < 0.5)),
            # 符号约定与论文/矩阵一致：gap = 跨来源均值 − 域内均值（崩塌时为负）。
            # 早期写成"域内 − 跨来源"得到 +0.41，会让下面的崩塌判定永远为假。
            "mean_gap": round(float(np.mean(crs) - np.mean(ind)), 4),
            "seconds": round(time.time() - t_all, 1),
        }
        s = entry["summary"]
        print(f"  → 域内均值 {s['in_domain_mean_auc']:.4f}  跨来源均值 {s['cross_mean_auc']:.4f}"
              f"  落差 {s['mean_gap']:+.4f}  低于随机 {s['n_below_chance']}/{s['n_cross_pairs']}"
              f"  ({s['seconds']:.0f}s)\n")
        report["probes"][kind] = entry

    # ---- 结论判定：各探针的定性结论是否一致
    gaps = {k: v["summary"]["mean_gap"] for k, v in report["probes"].items()}
    belows = {k: v["summary"]["n_below_chance"] for k, v in report["probes"].items()}
    all_collapse = all(g < -0.2 for g in gaps.values())
    incl = [k for k, v in report["probes"].items()
            if v["summary"].get("n_below_chance") is not None]
    below_frac = {k: (belows[k] / report["probes"][k]["summary"]["n_cross_pairs"])
                  for k in incl}
    report["verdict"] = {
        "mean_gap_by_probe": gaps,
        "below_chance_fraction_by_probe": {k: round(v, 3) for k, v in below_frac.items()},
        "all_probes_collapse": bool(all_collapse),
        "gap_range": [round(min(gaps.values()), 4), round(max(gaps.values()), 4)],
        "conclusion": (
            "跨来源崩塌在**全部被测探针上一致出现**，因此不是默认线性探针的产物"
            if all_collapse else
            "崩塌幅度随探针变化较大，需逐探针讨论"),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                      encoding="utf-8")
    (OUT / "results.json").read_text(encoding="utf-8")

    lines = ["# 探针稳健性：跨来源崩塌是否为默认线性探针的产物", "",
             f"编码器 `{args.encoder}`；来源 {len(names)} 个；"
             f"跨来源配对 {report['probes'][list(report['probes'])[0]]['summary']['n_cross_pairs']} 个。", "",
             "> 测试集始终为**全部**目标样本，AUC 在探针之间严格可比。",
             "> SVM / kNN 的训练集做分层子采样（上限 "
             f"{TRAIN_CAP['svm']}），实际训练量记录在 results.json 中。", "",
             "| 探针 | 说明 | 域内均值 AUC | 跨来源均值 AUC | 落差 | 低于随机 |",
             "| --- | --- | --- | --- | --- | --- |"]
    for k, v in report["probes"].items():
        s = v["summary"]
        lines.append(f"| `{k}` | {v['description']} | {s['in_domain_mean_auc']:.4f} | "
                     f"{s['cross_mean_auc']:.4f} | **{s['mean_gap']:+.4f}** | "
                     f"{s['n_below_chance']}/{s['n_cross_pairs']} |")
    v = report["verdict"]
    lines += ["", f"**结论**：{v['conclusion']}", "",
              f"- 各探针平均落差范围：**{v['gap_range'][0]:+.4f} 至 {v['gap_range'][1]:+.4f}**",
              f"- 全部探针均出现崩塌（落差 < −0.2）：**{v['all_probes_collapse']}**", ""]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(v, indent=2, ensure_ascii=False))
    print(f"\n-> {OUT/'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
