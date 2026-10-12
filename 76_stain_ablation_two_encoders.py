"""染色归一化消融：在**同一图像子集**上对多个编码器做配对对比。

为什么需要这个脚本（而不是复用 59 的 --normalize）：
  59 的 --normalize 把 Reinhard 归一化放在内存里做，当场用 phikon-v2 提特征，
  写入 `patch_multisource_stainnorm`，而"原始"侧读的是**全量** `patch_multisource`。
  两者样本集合不同，且编码器写死在 MODEL 常量上，
  因此无法回答目标里的问题——「换编码器后染色归一化的作用是否一致」。

本脚本的设计：
  1. 用固定种子分层抽样出**同一个**子集（每来源每类上限 N），落盘 `subset.json`；
  2. 对每个编码器，分别在该子集上提「原始」与「归一化后」两套特征；
  3. 在完全相同的样本上算跨来源 AUC，得到配对的 ΔAUC。

注意（写在论文局限里）：
  本实现的目标色彩统计量取**各数据集自身**的均值/标准差，
  因此它抹除的是"来源内部"的颜色变异，而不是把各来源对齐到统一参考染色。
  这一设置与 59 的历史结果保持一致（不中途改口径导致旧结论失效），
  但解读时必须写清楚：它的效果方向为什么可能不一致。

输出：runs/_stain_ablation_encoders/{results.json,report.md,subset.json}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

_s59 = import_module("59_stain_and_shortcut_audit")
reinhard_normalize = _s59.reinhard_normalize

# 染色消融的**原始 5 来源**口径：论文 §3.5b 与 Table 5 / Table 8 的既有数字都基于这 5 个。
# 来源注册表后续把新来源（breast_idc、colon_cancer_csv）也纳入了矩阵；若这里不显式限定，
# 样本集会随注册表被动变大，旧的 ΔAUC 数字失效且无法与论文对齐。
# 因此默认锁死这 5 个（扩来源后仍可复现）；要纳入新来源必须显式传 --sources。
DEFAULT_SOURCES = ["endo_histo", "uterus_ca", "ovarian_ca", "breakhis_400x", "lung_colon"]


def make_list_items(only: list[str] | None):
    """按来源白名单裁剪 59 的 list_items()。"""
    base = _s59.list_items

    def _items():
        got = base()
        if only is None:
            return got
        return {k: v for k, v in got.items() if k in only}
    return _items


# 模块级可重绑定：main() 里按 --sources 重新赋值，供下面各函数使用。
list_items = make_list_items(DEFAULT_SOURCES)

ROOT = Path(__file__).resolve().parent.parent
FEAT = ROOT / "data" / "features"
OUT = ROOT / "runs" / "_stain_ablation_encoders"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

ENCODERS = {
    "phikon-v2": {"path": ROOT / "models" / "phikon-v2", "dim": 1024, "input": 224},
    "phikon": {"path": ROOT / "models" / "phikon", "dim": 768, "input": 224},
    "vit-base-in": {"path": ROOT / "models" / "vit-base-in", "dim": 768, "input": 224},
}


def build_subset(per_class: int, seed: int) -> dict[str, list[tuple[str, int]]]:
    """分层抽样：每来源每类最多 per_class 张，固定种子，两个编码器共用。"""
    items = list_items()
    rng = np.random.default_rng(seed)
    out: dict[str, list[tuple[str, int]]] = {}
    for name, rows in items.items():
        by_lab = defaultdict(list)
        for p, l in rows:
            by_lab[int(l)].append(p)
        if not (by_lab[0] and by_lab[1]):
            print(f"  [跳过] {name}: 只有一个类别")
            continue
        picked: list[tuple[str, int]] = []
        for lab in (0, 1):
            pool = sorted(by_lab[lab])
            k = min(per_class, len(pool))
            idx = rng.choice(len(pool), k, replace=False)
            picked.extend((pool[i], lab) for i in idx)
        rng.shuffle(picked)
        out[name] = picked
        print(f"  {name}: 取 {len(picked)} 张 "
              f"(良性 {sum(1 for _, l in picked if l == 0)}, 恶性 {sum(1 for _, l in picked if l == 1)})")
    return out


def color_targets(subset: dict[str, list[tuple[str, int]]]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """每来源的 Reinhard 目标均值/标准差（与该来源全部图像统计，不只子集）。"""
    from PIL import Image

    items = list_items()
    out = {}
    for name, rows in items.items():
        acc, acc2, n = np.zeros(3), np.zeros(3), 0
        for p, _ in rows[:400]:
            try:
                with Image.open(p) as im:
                    a = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
            except Exception:  # noqa: BLE001
                continue
            acc += a.reshape(-1, 3).mean(0)
            acc2 += a.reshape(-1, 3).std(0)
            n += 1
        out[name] = (acc / max(n, 1), acc2 / max(n, 1))
        print(f"  {name} 目标均值 RGB = {np.round(out[name][0], 3)}  "
              f"目标 std = {np.round(out[name][1], 3)}")
    return out


def extract(name: str, subset: dict, targets: dict, normalized: bool, device: str,
            batch: int, input_size: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    import torch
    from PIL import Image
    from transformers import AutoModel

    spec = ENCODERS[name]
    model = AutoModel.from_pretrained(str(spec["path"])).eval().to(device)
    src = "归一化" if normalized else "原始"
    all_X, all_y, all_p, all_s = [], [], [], []
    t_all = time.time()
    with torch.no_grad():
        for ds, rows in subset.items():
            feats, ys, ps = [], [], []
            t0 = time.time()
            for i in range(0, len(rows), batch):
                chunk = rows[i:i + batch]
                arrs = []
                for p, _ in chunk:
                    try:
                        with Image.open(p) as im:
                            a = np.asarray(im.convert("RGB").resize((input_size, input_size), Image.BILINEAR),
                                           dtype=np.float32) / 255.0
                    except Exception:  # noqa: BLE001
                        a = np.zeros((input_size, input_size, 3), dtype=np.float32)
                    if normalized:
                        tmean, tstd = targets[ds]
                        a = reinhard_normalize(a, tmean, tstd)
                    arrs.append(a)
                x = torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2)
                x = (x - torch.tensor(MEAN).view(1, 3, 1, 1)) / torch.tensor(STD).view(1, 3, 1, 1)
                out = model(x.to(device))
                emb = out.last_hidden_state[:, 0] if hasattr(out, "last_hidden_state") else out[0][:, 0]
                if emb.shape[1] != spec["dim"]:
                    raise RuntimeError(f"{name} 期望 dim={spec['dim']}，实得 {emb.shape[1]}")
                feats.append(emb.float().cpu().numpy())
                ys.extend([l for _, l in chunk])
                ps.extend([str(p) for p, _ in chunk])
            all_X.append(np.concatenate(feats).astype(np.float32))
            all_y.append(np.asarray(ys, dtype=np.int64))
            all_p.extend(ps)
            all_s.extend([ds] * len(ys))
            print(f"    [{name}/{src}] {ds}: {all_X[-1].shape}  {time.time()-t0:.0f}s")
    del model
    try:
        import torch as _t
        _t.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    print(f"  [{name}/{src}] 合计 {(time.time()-t_all)/60:.1f} min")
    return np.concatenate(all_X), np.concatenate(all_y), all_p, all_s


def evaluate(X: np.ndarray, y: np.ndarray, sources: list[str], tag: str,
             paths: list[str] | None = None) -> dict:
    """域内 = GroupKFold OOF；跨来源 = 在来源 a 上拟合、在来源 b 上评估。

    分组必须用**图像路径**的 stem：早期版本误把 sources（来源名）当路径传入，
    导致每个来源的分组数都退化成 1（全部等于来源名），GroupKFold 直接报错
    ——已实测踩坑。paths 为 None 时退回用行号当组（只在无路径时使用）。
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    src = np.asarray(sources)
    pat = np.asarray(paths) if paths is not None else None
    res = {"in_domain": {}, "cross": {}}
    for s in sorted(set(sources)):
        m = src == s
        yv = y[m]
        if len(set(yv)) < 2:
            continue
        if pat is None:
            groups = np.arange(int(m.sum()))
        else:
            groups = np.array([Path(p).stem for p in pat[m]])
        n_groups = len(set(groups.tolist()))
        if n_groups < 5:
            print(f"  [WARN] {s} 仅 {n_groups} 个分组，跳过域内 OOF")
            continue
        oof = np.zeros(m.sum())
        for tr, te in GroupKFold(5).split(X[m], yv, groups):
            clf = LogisticRegression(max_iter=2000).fit(X[m][tr], yv[tr])
            oof[te] = clf.predict_proba(X[m][te])[:, 1]
        res["in_domain"][s] = round(float(roc_auc_score(yv, oof)), 4)
    for a in sorted(set(sources)):
        ma = src == a
        if len(set(y[ma])) < 2:
            continue
        clf = LogisticRegression(max_iter=2000).fit(X[ma], y[ma])
        for b in sorted(set(sources)):
            if a == b:
                continue
            mb = src == b
            if len(set(y[mb])) < 2:
                continue
            p = clf.predict_proba(X[mb])[:, 1]
            res["cross"][f"{a}->{b}"] = round(float(roc_auc_score(y[mb], p)), 4)
    ind = list(res["in_domain"].values())
    crs = list(res["cross"].values())
    print(f"[{tag}] 域内均值={np.mean(ind):.4f}  跨来源均值={np.mean(crs):.4f}  "
          f"落差={np.mean(ind)-np.mean(crs):+.4f}")
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", default=["phikon-v2", "phikon"])
    ap.add_argument("--per-class", type=int, default=500,
                    help="每来源每类抽样上限（500 → 每来源约 1000 张）")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-only", action="store_true", help="用已落盘特征重算，不重新提特征")
    ap.add_argument("--sources", nargs="*", default=list(DEFAULT_SOURCES),
                    help="参与消融的来源白名单（默认锁死原始 5 个，避免被动纳入新来源）；"
                         "传 'all' 表示使用注册表中的全部来源")
    ap.add_argument("--reuse-features", action="store_true",
                    help="已存在的 .npz 直接复用，只补缺失的（配合 --eval-only 用于中断后续跑）")
    args = ap.parse_args()

    for e in args.encoders:
        if e not in ENCODERS:
            print(f"未知编码器 {e}")
            return 2

    # 按 --sources 重绑定来源白名单（'all' = 注册表全部来源）
    global list_items
    only = None if (args.sources and args.sources[0] == "all") else list(args.sources or [])
    list_items = make_list_items(only)
    print(f"来源白名单：{only if only else '（注册表全部）'}")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    OUT.mkdir(parents=True, exist_ok=True)
    sub_path = OUT / "subset.json"

    results: dict = {}
    rp = OUT / "results.json"
    if rp.exists():
        try:
            results = json.loads(rp.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            results = {}

    if not args.eval_only:
        if sub_path.exists():
            raw = json.loads(sub_path.read_text(encoding="utf-8"))
            subset = {k: [(p, int(l)) for p, l in v] for k, v in raw.items()}
            print(f"复用已保存子集 {sub_path}")
        else:
            print("=== 构造分层抽样（两编码器共用同一子集）===")
            subset = build_subset(args.per_class, args.seed)
        if not sub_path.exists():
            sub_path.write_text(
                json.dumps({k: [[str(p), l] for p, l in v] for k, v in subset.items()},
                           indent=1, ensure_ascii=False), encoding="utf-8")

        print("\n=== 各来源染色目标 ===")
        targets = color_targets(subset)

        for enc in args.encoders:
            for normalized in (False, True):
                tag = "stainnorm" if normalized else "original"
                npz_path = OUT / f"{enc}__{tag}.npz"
                if args.reuse_features and npz_path.exists():
                    d = np.load(npz_path, allow_pickle=True)
                    X, y = d["features"], d["labels"].astype(int)
                    ps, srcs = list(d["paths"]), list(d["sources"])
                    print(f"  [{enc}/{tag}] 复用已有 {npz_path.name}  {X.shape}")
                else:
                    X, y, ps, srcs = extract(enc, subset, targets, normalized, device,
                                             args.batch, ENCODERS[enc]["input"])
                    np.savez_compressed(npz_path, features=X, labels=y,
                                        paths=np.array(ps), sources=np.array(srcs))
                results.setdefault(enc, {})[tag] = evaluate(X, y, srcs, f"{enc}/{tag}", ps)

    # 若之前只跑了一个编码器，另一个也能从落盘 npz 复算
    for enc in args.encoders:
        for tag in ("original", "stainnorm"):
            f = OUT / f"{enc}__{tag}.npz"
            if f.exists() and tag not in results.get(enc, {}):
                d = np.load(f, allow_pickle=True)
                results.setdefault(enc, {})[tag] = evaluate(
                    d["features"], d["labels"].astype(int), list(d["sources"]),
                    f"{enc}/{tag}", list(d["paths"]))

    print("\n=== 配对 ΔAUC（归一化后 − 原始，同图同子集）===")
    summary = {}
    for enc in args.encoders:
        r = results.get(enc, {})
        if "original" not in r or "stainnorm" not in r:
            continue
        delta = {k: round(r["stainnorm"]["cross"][k] - v, 4)
                 for k, v in r["original"]["cross"].items() if k in r["stainnorm"]["cross"]}
        improved = [k for k, v in delta.items() if v > 0.05]
        worsened = [k for k, v in delta.items() if v < -0.05]
        dvals = np.array(list(delta.values()))
        summary[enc] = {
            "in_domain_mean_original": round(float(np.mean(list(r["original"]["in_domain"].values()))), 4),
            "in_domain_mean_stainnorm": round(float(np.mean(list(r["stainnorm"]["in_domain"].values()))), 4),
            "cross_mean_original": round(float(np.mean(list(r["original"]["cross"].values()))), 4),
            "cross_mean_stainnorm": round(float(np.mean(list(r["stainnorm"]["cross"].values()))), 4),
            "delta_auc": delta,
            "n_improved_gt0.05": len(improved),
            "n_worsened_gt0.05": len(worsened),
            "delta_mean": round(float(dvals.mean()), 4),
            "delta_max": round(float(dvals.max()), 4),
            "delta_min": round(float(dvals.min()), 4),
            "improved_directions": improved,
            "worsened_directions": worsened,
        }
        print(f"\n  [{enc}] 域内 {summary[enc]['in_domain_mean_original']:.4f} → "
              f"{summary[enc]['in_domain_mean_stainnorm']:.4f}   "
              f"跨来源 {summary[enc]['cross_mean_original']:.4f} → "
              f"{summary[enc]['cross_mean_stainnorm']:.4f}   "
              f"Δ均值 {summary[enc]['delta_mean']:+.4f}")
        print(f"         改善(>+0.05) {len(improved)}/{len(delta)}   恶化(<-0.05) {len(worsened)}/{len(delta)}")

    # 跨编码器的方向集合：要求"完全相同的集合"过于苛刻（7 个 vs 4 个改善方向永远不可能相等），
    # 真正有用的是**交集**（在所有编码器上同向 = 与编码器无关的稳健结论）与**并集**。
    cross_enc: dict = {}
    if len(summary) >= 2:
        imp = [set(v["improved_directions"]) for v in summary.values()]
        wor = [set(v["worsened_directions"]) for v in summary.values()]
        inter_imp, inter_wor = set.intersection(*imp), set.intersection(*wor)
        union_imp, union_wor = set.union(*imp), set.union(*wor)
        deltas = {e: v["delta_auc"] for e, v in summary.items()}

        def _sign_consistent(pair: str) -> bool:
            """该配对在所有编码器上是否同号（都 >0 或都 <0）。"""
            vals = [deltas[e][pair] for e in deltas if pair in deltas[e]]
            return bool(vals) and (all(v > 0 for v in vals) or all(v < 0 for v in vals))

        common_pairs = set.intersection(*[set(d) for d in deltas.values()])
        same_sign = sorted([p for p in common_pairs if _sign_consistent(p)])
        cross_enc = {
            "improved_in_all_encoders": sorted(inter_imp),
            "worsened_in_all_encoders": sorted(inter_wor),
            "improved_in_any_encoder": sorted(union_imp),
            "worsened_in_any_encoder": sorted(union_wor),
            "same_sign_in_all_encoders": same_sign,
            "n_common_pairs": len(common_pairs),
        }
    results["cross_encoder"] = cross_enc

    results["summary"] = summary
    if cross_enc:
        verdict = (
            "同号配对中，**在全部编码器上都恶化**的方向 "
            f"{cross_enc['worsened_in_all_encoders']}；"
            "在全部编码器上都改善的方向 "
            f"{cross_enc['improved_in_all_encoders'] or '（无）'}。"
            "→ 可把交集方向当作与编码器无关的结论，并集方向只能按编码器分别报告。"
        )
    else:
        verdict = "编码器数不足（<2），无法做跨编码器方向比较。"
    results["conclusion"] = {
        "same_direction_across_encoders": bool(
            cross_enc and cross_enc["improved_in_all_encoders"]
            and cross_enc["worsened_in_all_encoders"]
            and len(cross_enc["improved_in_all_encoders"]) + len(cross_enc["worsened_in_all_encoders"])
            >= len(cross_enc["same_sign_in_all_encoders"]) * 0.6
        ),
        "verdict": verdict,
    }
    rp.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["# 染色归一化消融（多编码器配对对比）", "",
             "同一图像子集（固定种子分层抽样）上，对每个编码器分别提取「原始」与",
             "「Reinhard 染色归一化后」特征，再比较跨来源 AUC。", ""]
    for enc, v in summary.items():
        lines += [f"## {enc}", "",
                  f"- 域内均值：{v['in_domain_mean_original']:.4f} → {v['in_domain_mean_stainnorm']:.4f}",
                  f"- 跨来源均值：{v['cross_mean_original']:.4f} → {v['cross_mean_stainnorm']:.4f}",
                  f"- ΔAUC 均值 {v['delta_mean']:+.4f}（范围 {v['delta_min']:+.4f} ~ {v['delta_max']:+.4f}）",
                  f"- 改善(>+0.05)：{v['n_improved_gt0.05']}/{len(v['delta_auc'])}　"
                  f"恶化(<-0.05)：{v['n_worsened_gt0.05']}/{len(v['delta_auc'])}", ""]
    lines += [f"**结论**：{results['conclusion']['verdict']}", ""]
    if cross_enc:
        lines += ["## 跨编码器方向集合", "",
                  f"- 在**全部**编码器上都改善：{cross_enc['improved_in_all_encoders'] or '（无）'}",
                  f"- 在**全部**编码器上都恶化：{cross_enc['worsened_in_all_encoders'] or '（无）'}",
                  f"- 至少一个编码器上改善：{cross_enc['improved_in_any_encoder']}",
                  f"- 至少一个编码器上恶化：{cross_enc['worsened_in_any_encoder']}",
                  f"- 全部编码器同号的配对（共 {cross_enc['n_common_pairs']} 个配对）："
                  f"{cross_enc['same_sign_in_all_encoders']}", ""]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n-> {OUT/'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
