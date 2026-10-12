#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
65_verify_results_doc.py
========================
校验 docs/manuscript/Results_可直接引用.md 中引用的关键数字是否与 runs/ 下的
原始结果一致。防止手写稿件时把数字抄错。

做法：从原始 results.json 重新计算这些统计量，打印出来供人工比对。

用法：python scripts/65_verify_results_doc.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"


def load(p: Path) -> dict:
    if not p.exists():
        print(f"[MISS] {p}")
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _registry_source_count() -> int | None:
    """从来源注册表取当前来源数，用于检出'文档写 5 个来源、实际已 7 个'这类漂移。"""
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from source_registry import discover  # noqa: PLC0415
        return len(discover())
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    print("=" * 72)
    print("校验 Results 文档引用的数字")
    print("=" * 72)

    # ---- 3.2 / 3.3 跨来源矩阵
    m = load(RUNS / "_cross_source_matrix" / "results.json")
    if m:
        ind = m["in_domain"]
        cross = m["cross_matrix"]
        print("\n[3.2] 域内 AUC：")
        for k in sorted(ind, key=lambda x: -ind[x]["auc"]):
            print(f"   {k:16s} AUC={ind[k]['auc']:.4f}  AP={ind[k]['ap']:.4f}")
        aucs = [v["auc"] for v in cross.values()]
        below = [k for k, v in cross.items() if v["auc"] < 0.5]
        print(f"\n[3.3] 跨来源 {len(aucs)} 对：范围 {min(aucs):.4f}–{max(aucs):.4f}，"
              f"低于随机 {len(below)}/{len(aucs)} 对")
        print("   各目标域：")
        for k in sorted(m["as_target"], key=lambda x: m["as_target"][x]["mean_auc"]):
            t = m["as_target"][k]
            print(f"   {k:16s} 均值={t['mean_auc']:.4f} 最好={t['max_auc']:.4f} "
                  f"最差={t['min_auc']:.4f} 域内={t['in_domain_auc']:.4f} Δ={t['gap']:+.4f}")

    # ---- 3.4 捷径审计
    s = load(RUNS / "_stain_shortcut_audit" / "results.json")
    if s and "shortcut_audit" in s:
        sa = s["shortcut_audit"]
        print(f"\n[3.4] 捷径审计（{len(sa)} 个来源）：")
        vals = [sa[k]["stats_auc"] for k in sa]
        for k in sorted(sa, key=lambda x: -sa[x]["stats_auc"]):
            print(f"   {k:16s} AUC={sa[k]['stats_auc']:.4f}  "
                  f"n/类={sa[k]['n_per_class']}  最强={sa[k]['top_features'][0][0]}")
        print(f"   范围 {min(vals):.4f}–{max(vals):.4f}")
        # 不写死阈值：来源数变化后"全部 >x"这类断言会给出会过期的布尔值。
        # 改为报出低于参考线的来源，让读者自己判断。
        low = {k: v["stats_auc"] for k, v in sa.items() if v["stats_auc"] <= 0.79}
        print(f"   低于 0.79 的来源：{low if low else '无'}")
        n_reg = _registry_source_count()
        if n_reg is not None and n_reg != len(sa):
            print(f"   [!] 来源数不一致：注册表 {n_reg} 个 vs 捷径审计 {len(sa)} 个 "
                  f"→ 文档里关于'来源数/组合数'的表述需更新")

    # ---- 3.5 染色归一化
    if s and "cross_auc_delta_after_normalization" in s:
        d = s["cross_auc_delta_after_normalization"]
        vals = list(d.values())
        pos = [k for k, v in d.items() if v > 0]
        neg = [k for k, v in d.items() if v < 0]
        print(f"\n[3.5] 染色归一化（{len(d)} 对）：")
        print(f"   改善 {len(pos)}/{len(d)}，恶化 {len(neg)}/{len(d)}")
        kmax = max(d, key=lambda k: d[k]); kmin = min(d, key=lambda k: d[k])
        print(f"   最大改善 {kmax}: {d[kmax]:+.4f}")
        print(f"   最大恶化 {kmin}: {d[kmin]:+.4f}")
        o, n = s["eval_original"]["in_domain"], s["eval_stainnorm"]["in_domain"]
        print("   域内 AUC 变化（原始 -> 归一化）：")
        for k in sorted(o):
            print(f"     {k:16s} {o[k]:.4f} -> {n.get(k, float('nan')):.4f}  "
                  f"Δ={n.get(k, float('nan')) - o[k]:+.4f}")
        mean_delta = float(np.mean([n.get(k, np.nan) - o[k] for k in o]))
        print(f"   域内平均 Δ = {mean_delta:+.6f}")

    # ---- 3.6 外部活检
    b = load(RUNS / "_external_biopsy" / "results.json")
    if b:
        ag = b["aggregate"]
        print("\n[3.6] 外部活检域（多种子）：")
        for k, v in ag.items():
            print(f"   {k:28s} mean={v['mean']:.4f} sd={v['std']:.4f} "
                  f"range=[{v['min']:.4f},{v['max']:.4f}]")
        sens = ag["sensitivity_at_0.5"]["mean"]
        conf = ag["biopsy_mean_pred"]["mean"] - ag["source_malignant_mean_pred"]["mean"]
        print(f"   置信度落差 Δ = {conf:+.4f}")
        # 离散度基准要用「源域测试集」的离散度（来自跨标本类型实验），
        # 而不是源域恶性预测的离散度——两者差异很大，文档里必须写明用了哪个。
        sp_tgt = ag["biopsy_std_pred"]["mean"]
        sp_src_test = None
        xdoc = load(RUNS / "_cross_specimen" / "calibration_gap.json")
        if xdoc:
            sp_src_test = xdoc.get("stratified_by_label", {}).get("malignant", {}).get("spread_in")
        sp_src_mal = b["per_seed"][0]["source_malignant_std_pred"]
        print(f"   活检离散度 = {sp_tgt:.4f}")
        print(f"   源域恶性预测离散度 = {sp_src_mal:.4f}  → 倍数 {sp_tgt/max(sp_src_mal,1e-9):.2f}×")
        if sp_src_test:
            print(f"   源域测试集离散度 = {sp_src_test:.4f}  → 倍数 {sp_tgt/max(sp_src_test,1e-9):.1f}×"
                  f"  ← 文档引用的是这个基准")

    # ---- 3.8 校准
    c = load(RUNS / "_calibration" / "results.json")
    if c:
        print("\n[3.8] 校准迁移：")
        for k, v in c["summary"].items():
            if isinstance(v, (int, float)) and ("ece" in k or "delta" in k):
                print(f"   {k:44s} {v}")

    # ---- 3.7 跨标本类型
    x = load(RUNS / "_cross_specimen" / "calibration_gap.json")
    if x:
        print("\n[3.7] 跨标本类型校准差距：")
        st = x.get("stratified_by_label", {}).get("malignant", {})
        print(f"   域内 n={st.get('n_in')} 均值={st.get('mean_pred_in')}")
        print(f"   跨域 n={st.get('n_cross')} 均值={st.get('mean_pred_cross')}")
        print(f"   Δ={st.get('delta_mean_pred')}  置换检验 p={st.get('permutation_p')}")
        print(f"   ECE 比 = {x.get('ece_difference',{}).get('ratio')}")
        print(f"   离散度 in={st.get('spread_in')} cross={st.get('spread_cross')}")

    # ---- 3.5b 染色消融受控重做（双编码器同子集配对）
    sa = load(RUNS / "_stain_ablation_encoders" / "results.json")
    if sa:
        print("\n[3.5b] 染色归一化消融（受控重做，同子集配对）：")
        for enc, s in (sa.get("summary") or {}).items():
            print(f"   {enc:11s} 域内 {s['in_domain_mean_original']:.4f}→{s['in_domain_mean_stainnorm']:.4f}  "
                  f"跨来源 {s['cross_mean_original']:.4f}→{s['cross_mean_stainnorm']:.4f}  "
                  f"Δ均值 {s['delta_mean']:+.4f}  "
                  f"改善 {s['n_improved_gt0.05']}/20 恶化 {s['n_worsened_gt0.05']}/20")
        print(f"   方向跨编码器一致 = {sa.get('conclusion', {}).get('same_direction_across_encoders')}")
        # 交叉验证：子集上的原始落差应接近全量矩阵落差
        for enc, s in (sa.get("summary") or {}).items():
            gap_sub = s["in_domain_mean_original"] - s["cross_mean_original"]
            print(f"   子集原始落差 {enc:11s} = {gap_sub:+.4f}  ← 应与全量矩阵落差吻合")

    # ---- 3.7b 跨标本类型的第二编码器
    x2 = load(RUNS / "_cross_specimen_phikon" / "calibration_gap.json")
    if x2:
        st2 = x2.get("stratified_by_label", {}).get("malignant", {})
        print("\n[3.7b] 跨标本类型（第二编码器 phikon）：")
        print(f"   域内均值={st2.get('mean_pred_in')}  跨域均值={st2.get('mean_pred_cross')}")
        print(f"   Δ={st2.get('delta_mean_pred')}  置换检验 p={st2.get('permutation_p')}")
        print(f"   ΔCI95={st2.get('ci95_delta')}")

    # ---- 编码器对照汇总
    ec = load(RUNS / "_encoder_comparison" / "results.json")
    if ec:
        print("\n[汇总] 跨实验的编码器依赖：")
        v = ec.get("verdict", {})
        print(f"   全部编码器均崩塌 = {v.get('all_encoders_collapse')}  落差范围 {v.get('gap_range')}")
        print(f"   活检敏感度依赖编码器 = {v.get('biopsy_sensitivity_encoder_dependent')}")
        print(f"   活检敏感度 = {v.get('biopsy_sensitivity_by_encoder')}")
        for enc, rec in (ec.get("other_experiments") or {}).items():
            b, s, x3 = rec.get("biopsy"), rec.get("stain"), rec.get("cross_specimen")
            print(f"   {enc:11s} 染色Δ={s['delta_mean'] if s else 'n/a'}  "
                  f"活检敏感度={b['sensitivity_at_0.5'] if b else 'n/a'}  "
                  f"跨标本Δ概率={(x3 or {}).get('stratified_delta_mean_pred', 'n/a')}")

    # ---- 过期数字护栏：文档里是否还残留上一代（5 来源）的数字
    # 扩来源后最容易出的错是"算出了新数字，但文档某处还写着旧数字"。
    # 这里扫描三份成稿文件，报出与当前结果不一致的旧值。
    stale_checks = [
        ("0.9925", "跨来源矩阵旧域内均值（5 来源口径，现为 0.9467）"),
        ("−0.4173", "旧平均落差（5 来源口径，现为 −0.4028）"),
        ("-0.4173", "旧平均落差（5 来源口径，现为 −0.4028）"),
        ("6/20", "旧低于随机计数（5 来源口径，现为 15/42）"),
        ("0.356–0.824", "旧跨来源范围（5 来源口径，现为 0.3132–0.8243）"),
        ("0.7958", "旧 endo_histo 捷径 AUC（现为 0.8013）"),
        ("0.9790", "旧 lung_colon 捷径 AUC（现为 0.9775）"),
        ("5/5 来源", "旧捷径审计来源数（现为 7/7）"),
        ("20 个组合", "旧跨来源组合数（现为 42）"),
    ]
    docs = [
        ROOT / "docs" / "manuscript" / "Results_可直接引用.md",
        ROOT / "docs" / "manuscript" / "Methods_可直接引用.md",
        ROOT / "docs" / "manuscript" / "Introduction_Discussion_可直接引用.md",
        ROOT / "README.md",
    ]
    # 刻意保留的历史对照（已在这些文件中显式标注为"5 来源口径"或"当时的值"）。
    # 豁免必须写明理由；若某天这些文件真的漏改了，删掉豁免即可恢复告警。
    stale_allow = {
        ("Results_可直接引用.md", "0.9925"),   # §3.3 自我纠正段落，明写"falls from 0.9925 to"
        ("Results_可直接引用.md", "−0.4173"),  # §3.5b 交叉验证，明写"5 来源全量矩阵"
        ("Results_可直接引用.md", "6/20"),     # §3.3 自我纠正段落，明写"rises from 6/20"
        ("README.md", "−0.4173"),             # F35 明写"当时 5 来源全量矩阵"
    }
    hits: list[str] = []
    for p in docs:
        if not p.exists():
            continue
        txt = p.read_text(encoding="utf-8")
        for needle, why in stale_checks:
            if needle in txt and (p.name, needle) not in stale_allow:
                hits.append(f"   {p.name}: 含 `{needle}` — {why}")
    print("\n[护栏] 过期数字扫描：")
    if hits:
        print("\n".join(hits))
        print("   → 若这些是**刻意保留的历史对照**，请加注并在 stale_allow 中登记；否则应更新。")
    else:
        print("   未发现未登记的上代遗留数字 ✅（4 处已登记的历史对照豁免）")

    print("\n" + "=" * 72)
    print("请将以上数值与 docs/manuscript/*.md 逐条比对")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
