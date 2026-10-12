"""打包**纯代码备份**（给学生用）。

与其它包的区别
--------------
| 包 | 内容 | 体积 | 给谁 |
| --- | --- | --- | --- |
| `BSPC_submission_*.zip` | 正文、图、表 | ~1.4 MB | 期刊 |
| `BSPC_weights/features.zip` | 权重与特征张量 | 2.4 GB | 复现者 |
| **`BSPC_code_*.zip`（本脚本）** | **只有代码** | ~0.3 MB | **学生 / 接手者** |
| `BSPC_给学生的投稿包_*.zip` | 投稿用 docx | ~1.4 MB | 投稿操作 |

本包只含**可移植的代码**，不含数据、权重、特征、依赖库。设计要点：

1. **不含 `.pylibs`（4.6 GB）**，改为给 `requirements.txt` + 可移植的 `run.ps1`。
   学生要自己装依赖。原 `run.ps1` 硬编码了本机的 `D:\\Softwares\\Anaconda\\python.exe`，
   直接给会跑不起来，因此这里重写为**自动探测解释器**的版本。
2. **README 从磁盘实际内容生成**，不手工维护——避免"README 说 96 个脚本、实际 95 个"
   这类漂移（本项目已因手工维护清单踩过多次）。
3. **归档脚本单独放 `archive/superseded_scripts/`**，并在 README 中标注，让学生
   看得见历史但不误用。

用法：
    python scripts/100_package_code.py
"""
from __future__ import annotations

import re
import sys
import zipfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / "dist"
ARCHIVE_SRC = ROOT / "archive" / "superseded_scripts"

# 明确的运行顺序（README 用的三段式）。这里只列**必须按顺序**的核心链路。
PIPELINE = [
    ("准备", [
        ("env_check.py", "环境自检（解释器版本、依赖是否齐全）"),
        ("00_build_gdc_manifest.py", "GDC API → TCGA-UCEC 全量 manifest"),
        ("01_gdc_download.py", "切片下载器（多线程、断点续传、MD5 校验）"),
        ("02_data_inventory.py", "本地资产盘点 → slide_master.csv"),
        ("11_fetch_models.py", "编码器权重下载（phikon-v2 / phikon / vit-base-in）"),
    ]),
    ("WSI 主链路（Stage 1–3）", [
        ("10_tiling.py", "**Stage 1** 组织分割 + patch 坐标索引"),
        ("12_extract_features.py", "**Stage 2** 冻结特征提取 → HDF5"),
        ("13_mil_train_eval.py", "**Stage 3** MIL 训练 + AUC/CI/ECE"),
        ("14_build_split_manifest.py", "患者级划分（防泄漏）"),
        ("19_run_tcga_baseline.py", "一键跑通 TCGA 端到端基线"),
    ]),
    ("跨来源矩阵与消融", [
        ("source_registry.py", "**来源注册表（单一事实来源，改来源只改这里）**"),
        ("59_stain_and_shortcut_audit.py", "染色统计 + 捷径审计"),
        ("60_cross_source_matrix.py", "跨来源矩阵（提取 / 评估）"),
        ("73_encoder_comparison.py", "三编码器对照汇总"),
        ("76_stain_ablation_two_encoders.py", "染色归一化配对消融"),
    ]),
    ("独立性与稳健性检验", [
        ("91_shortcut_residualization.py", "捷径残差化（量化捷径贡献下界）"),
        ("92_probe_robustness.py", "探针稳健性（4 类探针同折划分）"),
        ("93_trivial_baselines.py", "平凡预测器基线"),
    ]),
    ("成稿与校验", [
        ("64_make_figures_tables.py", "生成 Figure 1–4 / Table 3–6"),
        ("80_regen_supplement_tables.py", "生成补充材料 S2–S11"),
        ("81_verify_supplement_numbers.py", "校验补充材料数字 ↔ JSON"),
        ("65_verify_results_doc.py", "校验正文数字 ↔ JSON（含过期护栏）"),
        ("87_assemble_manuscript.py", "合并各章 → Manuscript_Full.md"),
        ("88_package_submission.py", "打包投稿件（含 UTF-8 校验）"),
    ]),
]

# 数据摆放位置（学生必须自己下载并放到这些路径）
DATA_LAYOUT = [
    ("data/raw/patch_multisource/<dataset>/", "9 个 patch 来源图像（registry 里的 `root`）"),
    ("data/raw/endoaid_external/extracted/", "ENDO-AID 91 张活检 TIF"),
    ("data/raw/tcga_ucec_01z/", "TCGA-UCEC FFPE 诊断切片 01Z"),
    ("data/raw/tcga_ucec_normals/", "TCGA-UCEC 正常组织 11A"),
    ("models/<encoder>/", "编码器权重（config.json + model.safetensors）"),
    ("data/features/", "（可选）已有特征 npz/h5，放进即可跳过提特征"),
    ("runs/", "（自动创建）所有实验输出与 results.json"),
]


def collect_scripts() -> list[Path]:
    return sorted(p for p in (ROOT / "scripts").glob("*.py"))


def brief(p: Path) -> str:
    """抽出脚本的一句话说明。

    这些脚本的 docstring 结构是：
        #!/usr/bin/env python
        # -*- coding: utf-8 -*-
        \"\"\"
        00_build_gdc_manifest.py      <- 第一行是文件名
        ========================      <- 第二行是分隔线
        从 GDC API 拉取 …             <- 真正的说明在第三行
    早期实现只跳过以 `#`/`\"\"\"` 开头的行，于是抓到了**文件名本身**，
    自动生成的说明列全是文件名，毫无信息量（实测 97 行全是文件名）。
    """
    skip_exact = {p.name, "#!/usr/bin/env python", "# -*- coding: utf-8 -*-"}
    seps = set("=-~*#_ ")
    for ln in p.read_text(encoding="utf-8", errors="replace").splitlines()[:16]:
        s = ln.strip().strip('"').strip("'").strip()
        if not s:
            continue
        if s in skip_exact:
            continue
        if set(s) <= seps:                    # 纯分隔线
            continue
        # 有些脚本不用 docstring，而是 `# 说明` 形式（如 77_extract_vitbasein_wsi.py）。
        # 早期实现把 `#` 开头的行全部跳过，导致这些脚本的说明列退化成文件名。
        if s.startswith("#"):
            s = s.lstrip("#").strip()
            if not s or s.startswith(("!", "-*-", "用法", "python ", "为什么")):
                continue
        if s.startswith(("from ", "import ", ")")):
            continue
        return s.replace("|", "/")[:78]
    return ""


def readme_text() -> str:
    scripts = collect_scripts()
    archived = sorted(ARCHIVE_SRC.glob("*.py")) if ARCHIVE_SRC.exists() else []
    names = {p.name for p in scripts}

    lines = [
        "# 代码备份：跨来源病理基础模型稳健性基准",
        "",
        f"> 生成日期：{date.today().isoformat()}；由 `scripts/100_package_code.py` **从磁盘自动生成**。",
        "> 本包**只有代码**：不含数据、权重、特征、依赖库。数据需按下文自行下载。",
        "",
        "## 1. 这个项目在做什么（一句话）",
        "",
        "把三个**冻结**的病理基础模型编码器（`phikon-v2`、`phikon`、`vit-base-in`）",
        "在 **9 个公开 patch 数据集**上做完整的 9×9 跨来源迁移矩阵（72 个配对），",
        "并审计评测协议本身：域内高 AUC 有多少来自非形态学捷径、崩塌是否只是染色差异、",
        "以及校准与固定阈值行为在临床使用上是否可用。",
        "",
        "**关键设计**：编码器全程冻结，下游只用线性探针与小型 ABMIL，",
        "因此观察到的效应可归因于**表征**而非模型容量。",
        "",
        "## 2. 包里有什么",
        "",
        f"- `scripts/`：**{len(scripts)} 个 Python 脚本**（全部可移植，路径基于脚本自身位置解析）",
        f"- `archive/superseded_scripts/`：{len(archived)} 个**已归档**脚本（历史记录，勿用）",
        "- `README_CODE.md`：本文件",
        "- `requirements.txt`：依赖清单（需自行 `pip install`）",
        "- `run.ps1`：运行入口（自动探测 Python 解释器）",
        "- `PYTHON_RUNTIME.md`：解释器与编码注意事项",
        "",
        "## 3. 快速开始",
        "",
        "```powershell",
        "# 1) 建一个 Python 3.10+ 环境（3.12 最稳）",
        "python -m venv .venv",
        ".\\.venv\\Scripts\\Activate.ps1",
        "pip install -r requirements.txt",
        "",
        "# 2) 自检",
        ".\\run.ps1 scripts\\env_check.py",
        "",
        "# 3) 按 README_CODE 第 5 节的顺序跑",
        "```",
        "",
        "> ⚠️ **中文与编码**：源码含中文注释，全部为 UTF-8。",
        "> 运行前设 `$env:PYTHONUTF8 = '1'`（`run.ps1` 已自动设置）。",
        "> 写入文档**不要**用 PowerShell 的 `Set-Content` / `Add-Content` / `>`——",
        "> 它们会写出 ANSI/UTF-16 并**不可逆地损坏**中文。用 Python 以 `encoding='utf-8'` 写。",
        "",
        "## 4. 数据要放在哪里",
        "",
        "所有脚本用 `Path(__file__).resolve().parents[1]` 定位仓库根，因此**目录结构必须如下**：",
        "",
        "```",
        "跨域子宫内膜论文/",
        "├── scripts/                 ← 本包内容",
        "├── models/                  ← 权重（11_fetch_models.py 自动下载）",
        "├── data/",
        "│   ├── raw/",
        "│   │   ├── patch_multisource/   ← 9 个 patch 来源",
        "│   │   ├── endoaid_external/    ← ENDO-AID 活检",
        "│   │   ├── tcga_ucec_01z/       ← TCGA FFPE 诊断",
        "│   │   └── tcga_ucec_normals/   ← TCGA 正常组织",
        "│   ├── features/            ← 提取后的特征（npz / h5）",
        "│   └── manifests/           ← 清单 JSON",
        "└── runs/                    ← 实验输出（自动创建）",
        "```",
        "",
        "| 路径 | 放什么 |",
        "| --- | --- |",
    ]
    for path, what in DATA_LAYOUT:
        lines.append(f"| `{path}` | {what} |")

    lines += [
        "",
        "**9 个 patch 来源的确切定义写在 `scripts/source_registry.py` 里**，",
        "包括每个来源的下载位置、标签映射与分组方式。**这是唯一的来源事实来源**，",
        "要加/改来源只改这一个文件——`59` 与 `60` 都从它读取。",
        "",
        "## 5. 运行顺序",
        "",
        "> 下面只列**核心链路**。完整脚本清单见第 6 节。",
        "",
    ]
    for section, items in PIPELINE:
        lines += [f"### {section}", "",
                  "| 脚本 | 作用 |", "| --- | --- |"]
        for name, desc in items:
            mark = "" if name in names else " ⚠️**不在包内**"
            lines.append(f"| `scripts/{name}` | {desc}{mark} |")
        lines.append("")

    lines += [
        "## 6. 全部脚本（按编号）",
        "",
        f"包内共 {len(scripts)} 个。下表由脚本文件名自动生成，无手工维护。",
        "",
        "| 脚本 | 首行说明 |",
        "| --- | --- |",
    ]
    for p in scripts:
        lines.append(f"| `{p.name}` | {brief(p)} |")

    if archived:
        lines += [
            "",
            "## 7. 已归档脚本（**勿用**，仅作历史记录）",
            "",
            "| 归档脚本 | 为什么归档 |",
            "| --- | --- |",
        ]
        for p in archived:
            lines.append(f"| `{p.name}` | {brief(p)} |")
        lines += [
            "",
            "> `66`–`69` 是清理 S-BIAD1199 那批已废弃路线用的**一次性脚本**，",
            "> 已执行完毕；保留是为了让接手者看清「哪些路线被放弃、为什么」。",
        ]

    lines += [
        "",
        "## 8. 接手前必须知道的几件事（血泪教训）",
        "",
        "1. **不要在不同 `--segments` 之间复用分片下载目录**。`39_fetch_kaggle_zip.py` 已加拼装前校验：",
        "   若段文件与远端字节数不匹配会**拒绝拼装并报出精确差值**，而不是产出一个损坏文件。",
        "2. **不要用 PowerShell 重定向写文档**（见第 3 节）。",
        "3. **`59` 与 `60` 的来源集合必须一致**，否则捷径审计与矩阵无法互相印证。",
        "   这就是 `source_registry.py` 存在的原因——**永远不要在其中之一里硬编码来源表**。",
        "4. **`.pylibs` 不是缓存**：原环境里 Anaconda 基环境**没有** torch / transformers，",
        "   它们只存在于 `.pylibs`。本包不含它，请务必用 `requirements.txt` 自建环境。",
        "5. **穷尽搜索而非猜测**：本项目多次因「看起来合理」的猜测出错（例如把 `vit-base-in`",
        "   的上游仓库写成猜测值，实际是 `google/vit-base-patch16-224`）。写文档时宁可先查代码。",
        "",
        "## 9. 复现边界",
        "",
        "- **可复现**：给定 `data/features/` 下的特征，所有正文与补充材料数字都能由脚本重算。",
        "- **需重新下载**：原始图像（TCGA GDC / Zenodo / Kaggle）与编码器权重，合计 100 GB 量级。",
        "- **无不可复现部分**：全部依赖公开权重与公开数据。",
        "",
    ]
    return "\n".join(lines)


def requirements_text() -> str:
    """从实际 import 中提取依赖，避免凭记忆写 requirements。"""
    mods: set[str] = set()
    for p in collect_scripts():
        t = p.read_text(encoding="utf-8", errors="replace")
        for m in re.finditer(r"^\s*(?:import|from)\s+([A-Za-z_][A-Za-z0-9_]*)", t, re.M):
            mods.add(m.group(1))
    stdlib = {"__future__", "os", "sys", "re", "json", "time", "math", "csv", "glob",
              "shutil", "argparse", "dataclasses", "pathlib", "collections", "hashlib",
              "subprocess", "itertools", "functools", "typing", "warnings", "random",
              "zipfile", "urllib", "io", "datetime", "textwrap", "concurrent", "tempfile",
              "traceback", "statistics", "pickle", "base64", "unicodedata", "string",
              "contextlib", "importlib", "threading", "queue", "copy", "enum", "abc",
              "struct", "gzip", "tarfile", "socket", "xml", "html", "difflib", "logging",
              "platform", "shlex", "signal", "uuid", "zlib", "codecs", "operator",
              "decimal", "fractions", "bisect", "heapq", "secrets", "sched", "sqlite3"}
    third = sorted(m for m in mods if m not in stdlib and not m.startswith("_"))
    # ⚠️ 排除**本地模块**：`source_registry`、`pipx` 等是 scripts/ 下的同目录文件，
    # 不是 pip 包。若写进 requirements，`pip install` 会去 PyPI 找同名包——
    # `source_registry` 找不到会直接失败，而 `pipx` **恰好是 PyPI 上真实存在的另一个
    # 项目**，装了也用不了，且会把 `scripts/pipx.py` 遮蔽掉（实测踩到）。
    local_modules = {p.stem for p in collect_scripts()}
    third = [m for m in third if m not in local_modules]
    # 映射：import 名 -> pip 包名
    pip = {"sklearn": "scikit-learn", "PIL": "pillow", "cv2": "opencv-python",
           "yaml": "pyyaml", "fitz": "pymupdf", "pymupdf": "pymupdf",
           "h5py": "h5py", "torch": "torch", "torchvision": "torchvision",
           "transformers": "transformers", "numpy": "numpy", "pandas": "pandas",
           "matplotlib": "matplotlib", "scipy": "scipy", "tqdm": "tqdm",
           "openslide": "openslide-python", "safetensors": "safetensors",
           "huggingface_hub": "huggingface_hub", "requests": "requests",
           "docx": "python-docx", "seaborn": "seaborn"}
    lines = [
        "# 依赖清单（由 scripts/100_package_code.py 扫描实际 import 生成）",
        "# 装法：pip install -r requirements.txt",
        "# 注意：torch 请按自己的 CUDA 版本从 https://pytorch.org 选对应命令安装，",
        "#       本文件给出的是 CPU/通用默认。",
        "",
    ]
    for m in third:
        lines.append(pip.get(m, m))
    lines += [
        "",
        "# 本地模块（**不要** pip install，它们在 scripts/ 目录内）：",
        "#   source_registry.py   —— 9 个来源的注册表（59/60/65/91/92 都从它 import）",
        "#   pipx.py              —— 带读超时的下载器（17_download_uni2h_needed.py 用）",
        "#   env_check.py         —— 环境自检",
        "#   check_fonts.py       —— 绘图字体检查",
        "# 这些脚本开头会把自己所在目录加入 sys.path，因此从仓库根运行即可。",
        "",
        "# 读取 WSI（.svs/.tif）需要 OpenSlide **原生库**，pip 包之外还要装：",
        "#   Windows: 下载 openslide-win64 并把 bin 目录加入 PATH",
        "#   Linux:   apt-get install openslide-tools",
        "# 另外 libarchive 与 idc_index 是可选路径（52/58 两个脚本）用的，",
        "# 不跑那两步可以不装。",
        "",
    ]
    return "\n".join(lines)


RUN_PS1 = r"""<#
run.ps1 —— 运行入口（可移植版，自动探测 Python 解释器）
=====================================================
与原始版本的区别：不再硬编码本机的 Anaconda 路径，改为按优先级探测。
用法：
    .\run.ps1 scripts\env_check.py
    .\run.ps1 scripts\60_cross_source_matrix.py --eval --encoder phikon-v2
#>
param(
    [Parameter(Mandatory = $true, ValueFromRemainingArguments = $true)]
    [string[]]$Args
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot

# 依次尝试：虚拟环境 -> PATH 中的 python -> py 启动器
$candidates = @(
    (Join-Path $root '.venv\Scripts\python.exe'),
    (Join-Path $root 'venv\Scripts\python.exe')
)
$py = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $py) {
    $c = Get-Command python -ErrorAction SilentlyContinue
    if ($c) { $py = $c.Source }
}
if (-not $py) { throw "找不到 Python。请先建虚拟环境：python -m venv .venv" }

# 若仓库内存在 .pylibs（原开发环境的依赖），自动挂上；学生环境通常没有，可忽略
$pylibs = Join-Path $root '.pylibs'
if (Test-Path $pylibs) { $env:PYTHONPATH = $pylibs }

# 源码含中文，强制 UTF-8，避免 Windows 默认 GBK 解码失败
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'

Write-Host "[run] python = $py" -ForegroundColor DarkGray
& $py @Args
exit $LASTEXITCODE
"""


def main() -> int:
    stamp = date.today().isoformat()
    out = OUTDIR / f"BSPC_code_{stamp}.zip"
    OUTDIR.mkdir(exist_ok=True)

    scripts = collect_scripts()
    archived = sorted(ARCHIVE_SRC.glob("*.py")) if ARCHIVE_SRC.exists() else []
    n = 0
    problems: list[str] = []

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("README_CODE.md", readme_text())
        zf.writestr("requirements.txt", requirements_text())
        zf.writestr("run.ps1", RUN_PS1)
        zf.writestr("PYTHON_RUNTIME.md", (ROOT / "PYTHON_RUNTIME.md").read_text(encoding="utf-8"))
        n += 4
        for p in scripts:
            zf.write(p, f"scripts/{p.name}")
            n += 1
        for p in archived:
            zf.write(p, f"archive/superseded_scripts/{p.name}")
            n += 1

    print(f"已打包 {out.relative_to(ROOT)}")
    print(f"  条目 {n} 个（脚本 {len(scripts)} + 归档 {len(archived)} + 说明 4）"
          f"，{out.stat().st_size/1024:.0f} KB")

    # 自检：包内每个文本文件必须是合法 UTF-8（中文注释多，这是最容易出问题的地方）
    bad = 0
    checked = 0
    with zipfile.ZipFile(out) as zf:
        for info in zf.infolist():
            if info.filename.endswith(("/", ".py", ".md", ".txt", ".ps1")):
                checked += 1
                try:
                    zf.read(info).decode("utf-8")
                except UnicodeDecodeError:
                    problems.append(f"非 UTF-8：{info.filename}")
                    bad += 1
    print(f"  UTF-8 校验：{checked} 个文本文件，失败 {bad}")
    for x in problems:
        print("   -", x)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
