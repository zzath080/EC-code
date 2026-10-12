"""仓库整理：分类清点 + 生成清单 + 安全清理。

设计原则
--------
1. **先只读清点，再清理**：`--audit` 只报告不动文件；`--clean` 才执行删除。
2. **只删"可证明可重建"的东西**：环境缓存、已解压的压缩包、中断下载的残段、
   同一权重的冗余格式。**原始数据与中间产物一律不自动删**——那需要项目人决定。
3. **删前逐项打印绝对路径与体积**，符合"删除前确认目标路径即预期路径"的要求。
4. 权重与特征生成 SHA256 清单（`data/manifests/asset_manifest.json`），
   使"整理后包"可校验完整性。

用法：
    python scripts/96_organize_repo.py --audit
    python scripts/96_organize_repo.py --clean --yes
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def dir_size(p: Path) -> int:
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} PB"


@dataclass
class Item:
    path: Path
    reason: str
    kind: str = "clean"          # clean | keep | decide
    note: str = ""


@dataclass
class Audit:
    clean: list[Item] = field(default_factory=list)
    decide: list[Item] = field(default_factory=list)

    def add_clean(self, path: Path, reason: str) -> None:
        if path.exists():
            self.clean.append(Item(path, reason))

    def add_decide(self, path: Path, reason: str, note: str = "") -> None:
        if path.exists():
            self.decide.append(Item(path, reason, kind="decide", note=note))


def audit() -> Audit:
    a = Audit()

    # ---------- 1. 环境缓存（可重建，非项目产物） ----------
    # ⚠️ `.pylibs` **不在可删列表内**：它不是缓存，而是本项目唯一的 torch / transformers
    # 来源——Anaconda 基环境里这两个包不存在（已实测）。删掉它，特征提取与 WSI 训练就
    # 无法复现，而"重新安装"需要联网下载数 GB。因此它属于必须保留的复现资产。
    a.add_decide(ROOT / ".pylibs",
                 "项目专用依赖库（**含基环境缺失的 torch / transformers**）——复现必需，"
                 "但体积大且与代码无关，投稿包中不含；是否删除由项目人决定")
    a.add_clean(ROOT / ".piptmp", "pip 临时目录（空/可重建）")
    a.add_clean(ROOT / "wheels", "已安装完成的 wheel 缓存（可重建）")
    a.add_clean(ROOT / "wheels2", "已安装完成的 wheel 缓存（可重建）")
    a.add_clean(ROOT / "node_modules", "npm 依赖（package.json 可重建；本项目无 JS 代码依赖）")
    a.add_clean(ROOT / ".workbuddy", "工具缓存")

    # ---------- 2. 已解压的压缩包（解压目录已存在则原包冗余） ----------
    pairs = [
        (ROOT / "data/raw/patch_multisource/histopathologic-cancer-detection.zip",
         ROOT / "data/raw/patch_multisource/pcam", "PCam 图像"),
        (ROOT / "data/raw/patch_multisource/gastric-cancer-histopathology-tissue-image-dataset.zip",
         ROOT / "data/raw/patch_multisource/gastric_cancer", "GCHTID 图像"),
        (ROOT / "data/raw/endoaid_external/endometrium-carcinoma-pipelle.rar",
         ROOT / "data/raw/endoaid_external/extracted", "ENDO-AID 活检图"),
    ]
    for zp, ex, why in pairs:
        if zp.exists() and ex.exists() and any(ex.rglob("*")):
            a.add_clean(zp, f"{why}已解压到 {ex.relative_to(ROOT)}，原压缩包冗余")

    # ---------- 3. 中断下载残留 ----------
    # `*.part` 要区分两类：
    #   (a) 分段临时目录（._segments 等）——段已汇总或已放弃，**可删**；
    #   (b) GDC 单文件 `.svs.part`——`01_gdc_download.py` 支持 Range 续传，
    #       删掉等于丢弃已下载的进度（实测 3 个文件 2.79 GB），**归入"需决定"**。
    seen: set[Path] = set()

    def _seen(p: Path) -> bool:
        rp = p.resolve()
        if rp in seen:
            return True
        seen.add(rp)
        return False

    for p in sorted((ROOT / "data").rglob("._segments")):
        if p.is_dir() and not _seen(p):
            a.clean.append(Item(p, "分段下载临时目录（各段已汇总或已放弃）"))
    for p in sorted((ROOT / "data").rglob("._kaggle_segments*")):
        if p.is_dir() and not _seen(p):
            a.clean.append(Item(p, "Kaggle 分段下载临时目录（拼装后应已清空）"))

    resume_parts = [p for p in sorted((ROOT / "data").rglob("*.svs.part"))
                    if "._segments" not in p.parts and not _seen(p)]

    # ---------- 4. 冗余权重格式（同一模型两份） ----------
    ph = ROOT / "models/phikon"
    if (ph / "model.safetensors").exists() and (ph / "pytorch_model.bin").exists():
        a.add_clean(ph / "pytorch_model.bin",
                    "与 model.safetensors 同一权重；safetensors 为现行格式，且加载代码只用它")

    # ---------- 5. 需项目人决定的大件 ----------
    # 注意：`data/raw` 与 `tcga_ucec_tumor_subset` 存在**嵌套**，若都列会把体积算两遍。
    # 这里只列顶层 `data/raw`，残段进度作为独立说明附在其后（体积不重复计）。
    a.add_decide(ROOT / "data/raw", "原始数据与下载缓存（可由各来源重新获取）")
    if resume_parts:
        a.decide.append(Item(
            ROOT / "data/raw/tcga_ucec_tumor_subset",
            f"其中 {len(resume_parts)} 个 GDC 单文件残段（{human(sum(p.stat().st_size for p in resume_parts))}）"
            f"**可续传**，删除将丢弃进度",
            kind="decide", note="（体积已含在上一条，不重复计）"))
    a.add_decide(ROOT / "archive", "历史实验归档（含未完成下载）——刻意保留作为设计沿革记录")
    a.add_decide(ROOT / "TCGA-UCEC", "早期 TCGA 下载目录（与 data/raw/tcga_ucec_* 可能重叠）")
    return a


LEGACY_HINTS = ("legacy", "_v1", "old", "bak", "archive")


def build_manifest() -> dict:
    """权重与派生特征的 SHA256 清单（用于校验整理后包）。"""
    out: dict = {"models": {}, "features": {}}
    for f in sorted((ROOT / "models").rglob("*")):
        if f.is_file() and f.suffix in (".safetensors", ".bin", ".json"):
            out["models"][str(f.relative_to(ROOT)).replace("\\", "/")] = {
                "bytes": f.stat().st_size,
                "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
            }
    for f in sorted((ROOT / "data/features").rglob("*.npz")):
        out["features"][str(f.relative_to(ROOT)).replace("\\", "/")] = {
            "bytes": f.stat().st_size,
            "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", action="store_true", help="只清点，不动文件")
    ap.add_argument("--clean", action="store_true", help="执行清理")
    ap.add_argument("--manifest", action="store_true", help="生成权重/特征 SHA256 清单")
    ap.add_argument("--yes", action="store_true", help="确认执行删除")
    args = ap.parse_args()

    if args.manifest:
        m = build_manifest()
        dest = ROOT / "data/manifests/asset_manifest.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
        assert dest.read_text(encoding="utf-8")
        print(f"清单 -> {dest}")
        print(f"  模型文件 {len(m['models'])} 个，特征 {len(m['features'])} 个")
        return 0

    a = audit()
    print("=" * 96)
    print("可安全清理（可证明可重建）")
    print("=" * 96)
    tot = 0
    for it in sorted(a.clean, key=lambda x: -dir_size(x.path)):
        s = dir_size(it.path)
        tot += s
        print(f"  {human(s):>10}  {it.path.relative_to(ROOT)}")
        print(f"              └ {it.reason}")
    print(f"\n  小计 {human(tot)}")

    print("\n" + "=" * 96)
    print("需项目人决定（本脚本不自动删）")
    print("=" * 96)
    t2 = 0
    for it in sorted(a.decide, key=lambda x: -dir_size(x.path)):
        s = dir_size(it.path)
        t2 += s
        print(f"  {human(s):>10}  {it.path.relative_to(ROOT)}")
        print(f"              └ {it.reason}")
    print(f"\n  小计 {human(t2)}")

    if not args.clean:
        print("\n（未加 --clean，未删除任何文件）")
        return 0
    if not args.yes:
        print("\n[拒绝] --clean 需同时给 --yes")
        return 1

    print("\n" + "=" * 96)
    print("执行清理")
    print("=" * 96)
    freed = 0
    for it in sorted(a.clean, key=lambda x: -dir_size(x.path)):
        p = it.path.resolve()
        # 删除前确认目标确实在仓库内，且就是本项声明的路径
        if ROOT.resolve() not in p.parents:
            print(f"  [跳过] 越出仓库：{p}")
            continue
        s = dir_size(p)
        print(f"  删除 {p}")
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)
        if not p.exists():
            freed += s
        else:
            print("    [警告] 未能删除")
    print(f"\n已释放 {human(freed)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
