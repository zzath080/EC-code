"""按 BSPC 要求把参考文献按**正文出现顺序**统一重排。

背景：
    正文引用最初按"主题"分配编号（Introduction 用 1–15、Methods 用 18–19），
    Discussion 又复用了这些编号。BSPC 要求**按出现顺序**编号，因此需要一次统一重排。

顺序的确定方式（顺序即稿件结构：Abstract -> Introduction -> Methods -> Results -> Discussion）：
    Abstract 与 Results **不含**文献引用，因此实际顺序 = Introduction 的出现顺序，
    再接 Methods 中首次出现的条目。本脚本的 OLD_TO_NEW 就是按此顺序人工核定后写死的，
    并在运行前用 `_check_order()` 重新推导一次，**推导结果与映射不一致就报错退出**，
    避免映射写错却静默生效。

范围规则：`[a–b]` 这类区间会被逐项展开（`[4–7]` -> 4,5,6,7），只有**每一个**成员都已映射时
才整体重写；否则原样保留并报警（宁可漏改，不可错改）。

用法：
    python scripts/82_renumber_references.py --dry-run
    python scripts/82_renumber_references.py
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANU = ROOT / "docs" / "manuscript"

DOCS = ["Introduction_Discussion_可直接引用.md", "Methods_可直接引用.md",
        "Results_可直接引用.md"]

# 按出现顺序（Introduction 在前，Methods 在后）确定的旧 -> 新映射。
# 本表由 `_check_order()` 从文档实际内容推导后人工核定——不要手改后跳过自检。
OLD_TO_NEW = {
    1: 1,    # Bray GLOBOCAN 2022
    2: 2,    # Mohammadi（子宫内膜 WSI 数据集）
    3: 3,    # Thiringer（扫描仪域偏移损害校准）
    4: 4, 5: 5, 6: 6, 7: 7,   # [4-7] 病理基础模型（UNI / CONCH / Virchow2 / Phikon-v2）
    21: 8,   # van den Berg（EC 基础模型性能平台期）
    9: 9,    # Geirhos（捷径学习）
    10: 10,  # Zech（混杂变量）
    11: 11, 12: 12,  # 11–13 Tellez / Jahanifar / Boschman（染色归一化的批评性证据）
    13: 13,
    14: 14, 15: 15,  # Campanella / Neidlinger（模型排名基准）
    17: 16,  # Zamanitajeddin（DG 算法 benchmark）
    18: 17,  # Aubreville（MIDOG，扫描仪域偏移标准化基准）
    16: 18,  # Gelton（ENDO-AID 数据）
    22: 19,  # Wagner（结直肠跨标本类型有解）—— Discussion 2.4
    19: 20,  # Ilse（ABMIL）—— Methods 2.8
    20: 21,  # Lu 2021（CLAM）—— Methods 2.8
}

CITE_RE = re.compile(r"\[(\d+(?:\s*[,\u2013-]\s*\d+)*)\]")


def parse_group(g: str) -> list[int] | None:
    """把 `4-7` / `4,5` / `11` 解析成编号列表；无法解析返回 None。"""
    out: list[int] = []
    for part in re.split(r"[,\u2013-]", g):
        part = part.strip()
        if not part.isdigit():
            return None
        out.append(int(part))
    return out


def expand(g: str) -> list[int]:
    """展开区间：`4-7` -> [4,5,6,7]；`11,12` -> [11,12]。"""
    if "\u2013" in g or "-" in g:
        nums = parse_group(g)
        if nums and len(nums) == 2:
            lo, hi = nums
            return list(range(min(lo, hi), max(lo, hi) + 1))
        return nums or []
    return [int(x) for x in g.split(",") if x.strip().isdigit()]


def render(nums: list[int]) -> str:
    """把编号列表渲染回引用串：连续 >=3 个用区间，否则逗号分隔。"""
    nums = sorted(set(nums))
    if not nums:
        return "[]"
    if len(nums) >= 3 and nums == list(range(nums[0], nums[-1] + 1)):
        return f"[{nums[0]}\u2013{nums[-1]}]"
    return "[" + ",".join(str(n) for n in nums) + "]"


def _check_order() -> None:
    """从文档实际内容重新推导出现顺序，与 OLD_TO_NEW 比对。"""
    seen: list[int] = []
    for f in DOCS:
        p = MANU / f
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8")
        # 跳过"投稿前仍需处理"等元说明段，避免把说明里的编号计入
        text = re.sub(r"^>.*$", "", text, flags=re.M)
        for m in CITE_RE.finditer(text):
            for n in expand(m.group(1)):
                if n in OLD_TO_NEW and n not in seen:
                    seen.append(n)
    derived = {old: i + 1 for i, old in enumerate(seen)}
    mism = {k: (derived.get(k), OLD_TO_NEW.get(k))
            for k in set(derived) | set(OLD_TO_NEW) if derived.get(k) != OLD_TO_NEW.get(k)}
    if mism:
        print("[ERROR] 推导顺序与 OLD_TO_NEW 不一致（旧号: 推导值 vs 表中值）：")
        for k, v in sorted(mism.items()):
            print(f"    {k}: {v[0]} vs {v[1]}")
        print("   请修正 OLD_TO_NEW 后重跑；本脚本不猜测。")
        sys.exit(3)
    print(f"顺序自检通过：{len(seen)} 个旧编号参与重排")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-order-check", action="store_true")
    args = ap.parse_args()

    if not args.skip_order_check:
        _check_order()

    total = 0
    for f in DOCS:
        p = MANU / f
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8")
        changed = [0]
        unmapped: set[str] = set()

        def repl(m: re.Match) -> str:
            nums = expand(m.group(1))
            if not nums or any(n not in OLD_TO_NEW for n in nums):
                unmapped.add(m.group(0))
                return m.group(0)
            changed[0] += 1
            return render([OLD_TO_NEW[n] for n in nums])

        new = CITE_RE.sub(repl, text)
        if unmapped:
            print(f"  [WARN] {f}: 未映射的引用 {sorted(unmapped)}（已原样保留）")
        if new != text:
            if not args.dry_run:
                p.write_text(new, encoding="utf-8")
                back = p.read_text(encoding="utf-8")
                assert back == new, f"{f} 写回不一致"
            print(f"  {f}: 重写 {changed[0]} 处引用")
        total += changed[0]

    print(f"{'[dry] 将' if args.dry_run else '已'}重写 {total} 处引用")
    return 0


if __name__ == "__main__":
    sys.exit(main())
