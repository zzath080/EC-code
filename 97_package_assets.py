"""整理权重与派生资产：生成校验清单、资产清单文档，并打包。

为什么要单独整理
----------------
投稿包（`dist/BSPC_submission_*.zip`，~1.3 MB）**只含文字与图表**，不含权重与特征。
但"能否复现"取决于两样东西：**冻结编码器的权重**与**提取后的特征张量**。
二者体积分别为 ~1.8 GB 与 ~0.8 GB，不适合塞进正文包，因此单独成包并附校验清单。

产出
----
1. `data/manifests/asset_manifest.json` —— 每个权重/特征文件的字节数与 SHA256
2. `docs/Asset_Inventory.md` —— 人读的资产清单（含来源、许可、用途、校验方式）
3. `dist/assets/BSPC_weights.zip` —— 三个编码器权重（safetensors 格式）
4. `dist/assets/BSPC_features.zip` —— 全部 npz 特征
5. `dist/assets/CHECKSUMS.sha256` —— 上面两个包的校验值

用法：
    python scripts/97_package_assets.py --manifest     # 只生成清单与文档
    python scripts/97_package_assets.py --package      # 再打资产包
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist" / "assets"
MANIFEST = ROOT / "data" / "manifests" / "asset_manifest.json"
DOC = ROOT / "docs" / "Asset_Inventory.md"

# 权重来源（用于人读清单）。仓库 ID 取自 `scripts/11_fetch_models.py` 的 OPEN_MODELS，
# **不凭记忆填写**——早期版本这里曾写成猜测值（"MahmoodLab/uni 之外的 ImageNet 对照"），
# 与实际的 `google/vit-base-patch16-224` 不符。
WEIGHT_SOURCES = {
    "phikon-v2": ("owkin/phikon-v2", "ViT-L/14, DINOv2 自监督, 160k+ WSI", "Apache-2.0"),
    "phikon": ("owkin/phikon", "ViT-B/16, iBOT 自监督, 40M TCGA tiles", "Apache-2.0"),
    "vit-base-in": ("google/vit-base-patch16-224",
                    "ViT-B/16, ImageNet-1k 监督（**非病理模型**，用作对照）", "Apache-2.0"),
}


def sha256(p: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


def collect() -> dict:
    out: dict = {"weights": {}, "features": {}, "generated_by": "scripts/97_package_assets.py"}
    for f in sorted((ROOT / "models").rglob("*")):
        if f.is_file():
            rel = str(f.relative_to(ROOT)).replace("\\", "/")
            out["weights"][rel] = {"bytes": f.stat().st_size, "sha256": sha256(f)}
    for f in sorted((ROOT / "data/features").rglob("*.npz")):
        rel = str(f.relative_to(ROOT)).replace("\\", "/")
        out["features"][rel] = {"bytes": f.stat().st_size, "sha256": sha256(f)}
    return out


def write_doc(m: dict) -> None:
    w_tot = sum(v["bytes"] for v in m["weights"].values())
    f_tot = sum(v["bytes"] for v in m["features"].values())
    lines = [
        "# 资产清单：冻结编码器权重与派生特征",
        "",
        "> 本文件由 `scripts/97_package_assets.py` 生成，**不手工维护**。",
        "> 正文投稿包（`dist/BSPC_submission_*.zip`）只含文字与图表；权重与特征单独成包，",
        "> 校验值见 `dist/assets/CHECKSUMS.sha256`。",
        "",
        f"- 权重：{len(m['weights'])} 个文件，合计 **{human(w_tot)}**",
        f"- 特征：{len(m['features'])} 个文件，合计 **{human(f_tot)}**",
        "",
        "## 1. 编码器权重（`models/`）",
        "",
        "| 编码器 | 架构与预训练 | 上游 | 许可 | 本地文件 | 大小 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for enc, (repo, arch, lic) in WEIGHT_SOURCES.items():
        files = {k: v for k, v in m["weights"].items() if f"/{enc}/" in k}
        main = [k for k in files if k.endswith(".safetensors")]
        sz = sum(files[k]["bytes"] for k in main)
        lines.append(f"| `{enc}` | {arch} | {repo} | {lic} | "
                     f"{', '.join(Path(k).name for k in sorted(main))} | {human(sz)} |")
    lines += [
        "",
        "**每个编码器目录的完整内容**（`config.json` 与 `preprocessor_config.json` 是加载所必需，",
        "体积虽小但缺一不可）：",
        "",
        "| 编码器 | 文件 | 大小 | SHA256（前 16 位） |",
        "| --- | --- | --- | --- |",
    ]
    for k in sorted(m["weights"]):
        parts = k.split("/")
        if len(parts) >= 3:
            lines.append(f"| `{parts[1]}` | `{parts[2]}` | {human(m['weights'][k]['bytes'])} | "
                         f"`{m['weights'][k]['sha256'][:16]}` |")
    lines += [
        "",
        "**格式说明**：三个编码器均以 `safetensors` 提供。`phikon` 原有的 `pytorch_model.bin`",
        "副本已删除——它与 safetensors 是同一权重，且 `transformers` 5.x 因 CVE-2025-32434",
        "拒绝加载该格式（`scripts/72_bin_to_safetensors.py` 记录了转换过程）。",
        "",
        "**`vit-base-in` 的定位**：它是 ImageNet-1k 监督的通用 ViT，**不是病理模型**，",
        "在本研究中作为「非病理预训练」对照。其 config 保留 1000 类 ImageNet 标签表，",
        "这是上游模型的原始状态，未被修改。",
        "",
        "**许可**：三者均为 Apache-2.0，允许再分发；本地副本未做任何权重修改。",
        "",
        "## 2. 派生特征（`data/features/`）",
        "",
        "每个 `.npz` 含三个数组：`features`（N×D 浮点）、`labels`（N 个 0/1）、",
        "`paths`（N 条来源图像路径）。**这些是全部实验的输入**，因此不必重新解码 WSI 即可",
        "复算正文与补充材料的每一个数字（见 `scripts/65_verify_results_doc.py`）。",
        "",
        "| 子目录 | 编码器 | 来源数 | 说明 |",
        "| --- | --- | --- | --- |",
    ]
    groups: dict[tuple[str, str], int] = {}
    for k in m["features"]:
        parts = k.split("/")
        enc, sub = parts[2], parts[3] if len(parts) > 3 else "-"
        groups[(enc, sub)] = groups.get((enc, sub), 0) + 1
    note = {"patch_multisource": "9 个跨来源 patch 数据集（主矩阵）",
            "patch_multisource_stainnorm": "同一来源的 Reinhard 归一化版本（染色消融）",
            "patch_multidataset": "早期两来源版本（保留作历史对照）",
            "endometrial_patch": "子宫内膜 patch（早期）"}
    for (enc, sub), n in sorted(groups.items()):
        lines.append(f"| `{sub}` | `{enc}` | {n} | {note.get(sub, '—')} |")
    lines += [
        "",
        "## 3. 校验方式",
        "",
        "```powershell",
        "# 逐文件校验（PowerShell 7 的 Get-FileHash 支持 sha256）",
        "Get-FileHash models/phikon-v2/model.safetensors -Algorithm SHA256",
        "# 与 data/manifests/asset_manifest.json 中的 sha256 字段比对",
        "```",
        "",
        "## 4. 复现边界（必须如实说明）",
        "",
        "- **可复现**：从上述特征出发，矩阵、捷径审计、染色消融、校准、探针稳健性、",
        "  平凡基线、残差化等全部数字，均可由 `scripts/` 重算。",
        "- **需重新下载**：原始 WSI 与 patch 图像（TCGA GDC、Zenodo、Kaggle），",
        "  体积 100 GB 级以上，已按 `scripts/96_organize_repo.py` 的规则清理冗余副本。",
        "- **不可复现的部分**：无。所有实验均只依赖公开权重与公开数据。",
        "",
    ]
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text("\n".join(lines), encoding="utf-8")
    assert DOC.read_text(encoding="utf-8")
    print(f"清单文档 -> {DOC.relative_to(ROOT)}")


def make_zip(dest: Path, files: list[tuple[Path, str]]) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED, compresslevel=6,
                         allowZip64=True) as zf:
        for src, arc in files:
            zf.write(src, arc)
    h = sha256(dest)
    print(f"  {dest.name}: {len(files)} 个文件，{human(dest.stat().st_size)}")
    return h


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", action="store_true")
    ap.add_argument("--package", action="store_true")
    args = ap.parse_args()
    if not (args.manifest or args.package):
        ap.print_help()
        return 0

    m = collect()
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    assert MANIFEST.read_text(encoding="utf-8")
    print(f"校验清单 -> {MANIFEST.relative_to(ROOT)}")
    print(f"  权重 {len(m['weights'])} 个，特征 {len(m['features'])} 个")
    write_doc(m)

    if not args.package:
        return 0

    w_files = [(ROOT / k, k) for k in m["weights"]]
    f_files = [(ROOT / k, k.replace("data/features/", "features/")) for k in m["features"]]
    print("\n打包资产：")
    hw = make_zip(DIST / "BSPC_weights.zip", w_files)
    hf = make_zip(DIST / "BSPC_features.zip", f_files)
    ck = DIST / "CHECKSUMS.sha256"
    ck.write_text(
        f"{hw}  BSPC_weights.zip\n{hf}  BSPC_features.zip\n",
        encoding="utf-8")
    assert ck.read_text(encoding="utf-8")
    print(f"  CHECKSUMS.sha256 已写入")
    return 0


if __name__ == "__main__":
    sys.exit(main())
