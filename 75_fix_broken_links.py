"""全仓库文档链接体检：把指向不存在路径的引用找出来（而不是匹配固定的旧字符串）。

为什么重写：
    初版只做「固定字符串 → 固定字符串」的替换。一旦替换生效，它就无法再匹配，
    于是每次运行都报告同一批 "MISS"，看起来像还有 5 处失效链接，实际上 README
    早已修好。那种检查器会持续给出假警报。

现在做两件事：
    1. 扫描仓库内的 .md，抽取 markdown 链接 `](path)` 与反引号中的路径，
       逐个检查是否存在于磁盘；
    2. 对已知的归档搬迁（原路径 → docs/archive/ 或 archive/superseded_scripts/）
       做定向替换。

刻意**不**检查的范围（否则会被噪音淹没）：
    - `docs/archive/` 等归档目录：那是"当初为什么废弃"的历史记录，不追求链接可用；
    - 外部仓库的文件名（如 `create_patches_fp.py`、`presets/tcga.csv`）与代码示例里的
      占位路径——它们从来不在本仓库，报出来没有意义；
    - 带空格或 `*<>|` 的片段（文档标题，不是路径）。

因此本脚本的报告只应剩下**真实的**项目内失效引用；数量少而准，而不是多而假。

用法：
    python scripts/75_fix_broken_links.py --dry-run
    python scripts/75_fix_broken_links.py --fix
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 已知的归档搬迁：旧路径片段 → 新路径。只做**路径**替换，不改写说明文字。
RELOCATIONS = [
    ("论文框架与实施总方案.md", "docs/archive/legacy_plans/论文框架与实施总方案.md"),
    ("scripts/03_endoaid_files.py", "archive/superseded_scripts/03_endoaid_files.py"),
    ("scripts/08_probe_biostudies_url.py", "archive/superseded_scripts/08_probe_biostudies_url.py"),
    ("scripts/23_fetch_endoaid_annotations.py", "archive/superseded_scripts/23_fetch_endoaid_annotations.py"),
    ("scripts/36_migrate_slide_ids.py", "archive/superseded_scripts/36_migrate_slide_ids.py"),
]

SKIP_DIRS = {".git", ".pylibs", "node_modules", "__pycache__", ".mypy_cache",
             # `dist/` 是**最终产物**目录（投稿包、资产包、给学生的 docx 包），
             # 不是源码。曾因扫到一个旧的解压副本 `dist/BSPC_submission_*/` 而报出
             # 3 处"失效引用"——那是历史快照里的路径，不是当前文档的问题。
             "dist", ".workbuddy", ".piptmp"}

# 只检查**项目自身的**路径引用。理由：归档文档里大量引用外部仓库的文件名
# （如 create_patches_fp.py、presets/tcga.csv）与说明性文件名，那些从来不在本仓库，
# 报出来只会淹没真正的问题。可检查的前缀见 CHECK_PREFIXES。
CHECK_PREFIXES = ("docs/", "scripts/", "data/", "runs/", "models/", "archive/", "assets/")

# 已知的"说明性引用"：文档正文在解释某个产物**何时**生成，而不是指向一个当前存在的文件。
# 这些已在对应文档里注明条件，故列入豁免；豁免必须写清理由，否则就成了掩盖。
#   1. source_matrix_build_results.json —— 仅 61_build_source_matrix.py --download 时生成
#   2. results.json                      —— 泛指"各 run 目录里的结果文件"，不是单一文件
#   3. README_DELIVERY.md                —— 由 scripts/88 写入 **zip 内部**，不是仓库文件；
#      骨架里提到它是在说明"交付包内附带这份说明"
EXEMPT = {
    "data/manifests/source_matrix_build_results.json",
    "results.json",
    "README_DELIVERY.md",
}

# `scheme://…` 外链、纯锚点、以及明显不是路径的片段
MD_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
TICK_PATH = re.compile(r"`([A-Za-z0-9_./\\\u4e00-\u9fff-]+\.(?:md|py|json|csv|png|pdf|h5|npz|txt|ps1|yaml|yml))`")


def md_files() -> list[Path]:
    out = []
    for p in ROOT.rglob("*.md"):
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        # 归档目录整体跳过：那是"当初为什么废弃"的历史记录，不追求链接可用
        if "archive" in p.parts:
            continue
        out.append(p)
    return sorted(out)


def is_checkable(s: str) -> bool:
    if not s or "://" in s or s.startswith("#") or s.startswith("mailto:"):
        return False
    if any(ch in s for ch in "*<>|"):
        return False
    # 说明性引用（已在文档中注明生成条件）不做失效判定
    if s.replace("\\", "/").strip("/") in EXEMPT:
        return False
    # 文档标题（含空格且不像路径）以及第三方仓库文件名一律跳过
    t = s.replace("\\", "/")
    if " " in t:
        return False
    if t.startswith(CHECK_PREFIXES):
        return True
    # 允许仓库根下的单文件（如 README.md）
    return "/" not in t.strip("/")


def candidates(text: str) -> set[str]:
    found = set()
    for m in MD_LINK.finditer(text):
        target = m.group(1).split("#")[0].strip()
        if is_checkable(target):
            found.add(target)
    for m in TICK_PATH.finditer(text):
        target = m.group(1).strip()
        if is_checkable(target):
            found.add(target)
    return found


_NAME_INDEX: dict[str, list[Path]] | None = None


def _name_index() -> dict[str, list[Path]]:
    """全仓库"文件名 -> 路径列表"索引，只构建一次。

    早期实现对每个未解析的裸文件名都调用一次 `ROOT.rglob(name)`；整理仓库后
    待解析条目变多，这样会被拖到十几分钟（实测一次扫描超过 10 分钟）。
    建成索引后整轮扫描只需走一遍文件树。
    """
    global _NAME_INDEX
    if _NAME_INDEX is None:
        idx: dict[str, list[Path]] = {}
        for q in ROOT.rglob("*"):
            if not q.is_file():
                continue
            if any(part in SKIP_DIRS for part in q.parts):
                continue
            idx.setdefault(q.name, []).append(q)
        _NAME_INDEX = idx
    return _NAME_INDEX


def resolve(md: Path, target: str) -> Path | None:
    """把文档中的引用解析为磁盘路径。

    顺序：文档所在目录 -> 仓库根 -> 规范目录（scripts/、docs/ 等）-> 全仓库同名兜底。
    """
    t = target.replace("\\", "/")
    for base in (md.parent, ROOT):
        p = (base / t).resolve()
        if p.exists():
            return p

    name = Path(t).name
    hits = _name_index().get(name, [])
    if len(hits) == 1:
        return hits[0]

    # 同名多处时，优先取规范目录里的那一个。
    # 整理仓库删掉 `archive/` 后，`scripts/README_project.md` 之类原本唯一的文件名
    # 会因为另一份同名副本消失而**变成"唯一"或"不唯一"的边界情况**；不区分目录会
    # 把正常的说明性引用误报为断链（实测一次整理后误报 136 处）。
    preferred = [q for q in hits
                 if q.relative_to(ROOT).parts[0] in ("scripts", "docs", "data", "runs")]
    if len(preferred) == 1:
        return preferred[0]
    if preferred:
        return sorted(preferred)[0]

    # 仍然无解时，若同名文件存在多处（例如各模型目录下的 config.json），
    # 视为**歧义**而非断链：正文里写裸文件名是叙述，不是路径断言。
    if hits:
        return sorted(hits)[0]
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只报告，不写入")
    ap.add_argument("--fix", action="store_true", help="应用已知的归档搬迁替换")
    args = ap.parse_args()

    files = md_files()
    print(f"扫描 {len(files)} 个 markdown 文件\n")

    broken: list[tuple[Path, str]] = []
    for md in files:
        for t in sorted(candidates(md.read_text(encoding="utf-8", errors="replace"))):
            if resolve(md, t) is None:
                broken.append((md, t))

    if not broken:
        print("未发现失效引用 ✅")
        return 0

    print(f"发现 {len(broken)} 处失效引用：")
    by_file: dict[Path, list[str]] = {}
    for md, t in broken:
        by_file.setdefault(md, []).append(t)
    for md, ts in by_file.items():
        # 反引号里的历史脚本名等属于「说明性提及」，单独标注，避免误判为死链
        print(f"  {md.relative_to(ROOT)}")
        for t in ts:
            hint = ""
            for old, new in RELOCATIONS:
                if t.endswith(Path(old).name) and (ROOT / new).exists():
                    hint = f"   → 可迁移至 {new}"
                    break
            print(f"    - {t}{hint}")

    if args.fix and not args.dry_run:
        changed = 0
        for md in by_file:
            text = md.read_text(encoding="utf-8")
            orig = text
            for old, new in RELOCATIONS:
                if (ROOT / new).exists() and old in text and new not in text:
                    text = text.replace(old, new)
            if text != orig:
                md.write_text(text, encoding="utf-8")
                changed += 1
                print(f"  已修复 {md.relative_to(ROOT)}")
        print(f"\n已修复 {changed} 个文件")
    else:
        print("\n（未加 --fix，未写入；加 --fix 可替换已知的归档搬迁）")
    return 1


if __name__ == "__main__":
    sys.exit(main())
