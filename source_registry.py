"""跨来源矩阵的**唯一**来源注册表。

为什么需要这个模块
------------------
`59_stain_and_shortcut_audit.py` 与 `60_cross_source_matrix.py` 原本各自硬编码了一份
来源表（`DATASETS` + `MALIG` 与 `SPECS`）。加一个来源要改两处，两处一旦不一致，
捷径审计与矩阵就在不同的来源集合上跑——而这两者是要互相印证的，不一致会让结论失去意义。
（这正是本项目已经踩过的坑：早期 `FEAT_ORIG` 指向 2 个来源、归一化特征有 5 个来源。）

因此把所有来源定义集中到这里，两个脚本都从这里取。

类别语义的统一约定
------------------
每个来源必须能折叠成「恶性（label=1） vs 良性/非恶性（label=0）」。
不同来源的原始类别定义不同（例如 `lung_colon` 是"正常上皮 vs 腺癌/鳞癌"，
`endo_histo` 是"正常/增生/息肉 vs 腺癌"），这属于已知局限，**必须在论文中写明**：
跨来源 AUC 因此不是纯粹的域偏移度量，而是"域偏移 + 类别定义差异"的复合量。

支持的标签来源（`label_from`）
-----------------------------
- `"dir"`（默认）：类别写在**目录名**里，与 `malignant` / `benign` 集合比对。
- `"filename"`：类别写在**文件名**里，用 `label_regex` 抽出组 1 再查 `filename_labels`。
  用于 Breast IDC 数据集（文件名 `...class0.png` / `...class1.png`）。
- `"csv"`：类别写在 **CSV** 里（列名 `csv_image_col` / `csv_label_col`），
  用于标签即使用说明的数据集（如 Colon-Cancer-datasets 的 `train.csv`）。
"""
from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MS = ROOT / "data" / "raw" / "patch_multisource"
EP = ROOT / "data" / "raw" / "endometrial_patch" / "extracted"

# 图像扩展名：**必须包含 .png**。历史 bug：`60` 早期只认 jpg/jpeg/tif/tiff，
# 导致 BreakHis（1693 张全是 .png）在计数与提取时被整个跳过而不报错。
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


SPECS: dict[str, dict] = {
    "endo_histo": {
        "root": EP / "HIENet",
        "malignant": {"EA"}, "benign": {"NE", "EH", "EP"},
        "organ": "endometrium",
        "note": "正常/增生/息肉 vs 腺癌",
    },
    "uterus_ca": {
        "root": EP / "Uterus-Cancer",
        "malignant": {"tumour"}, "benign": {"non-tumour"},
        "organ": "uterus",
        "note": "肿瘤 vs 非肿瘤",
    },
    "ovarian_ca": {
        "root": MS / "ovarian_ca",
        "malignant": {"Clear_Cell", "Endometri", "Mucinous", "Serous"},
        "benign": {"Non_Cancerous"},
        "organ": "ovary",
        "note": "非癌 vs 四种卵巢癌亚型",
    },
    "breakhis_400x": {
        "root": MS / "breakhis-400x",
        "malignant": {"malignant"}, "benign": {"benign"},
        "organ": "breast",
        "note": "乳腺良性 vs 恶性（400×，经典基准数据集）",
    },
    "lung_colon": {
        "root": MS / "lung_colon",
        "malignant": {"colon_aca", "lung_aca", "lung_scc"},
        "benign": {"colon_n", "lung_n"},
        "organ": "lung+colon",
        "note": "肺与结肠：正常上皮 vs 腺癌/鳞癌（5 类各 5000，折叠为二分类）",
    },
    # ---- 扩展来源 ----
    "breast_idc": {
        # 文件名格式 u_xX_yY_classC.png，C=0 非 IDC（**良性**）/ C=1 IDC（**恶性**）。
        # 目录结构 <patient_id>/class0|class1/*.png。
        # ⚠️ 标签方向易错：0 是"无 IDC"= 良性，1 是"有 IDC"= 恶性。写反会让整个来源
        #    的恶性/良性计数互换（实测踩过：写成 {0:1, 1:0} 得到 397476 恶性 / 157572 良性）。
        # 该来源的独特价值：**带患者 ID**，可做患者级分组，修正其余来源
        # "无患者标识、只能按图像编号分组"的局限。
        "root": MS / "breast-histopathology-images",
        "label_from": "filename",
        "label_regex": r"class(\d)",
        "filename_labels": {"0": 0, "1": 1},
        "organ": "breast",
        "note": "IDC 阳性（恶性）vs 阴性（良性），50×50 px；**带患者 ID**，可做患者级分组",
        # 患者 ID 嵌在文件名前缀（`10253_idx5_...`），不是目录名——目录名是 class0/class1
        "group_regex": r"^(.+?)_x\d+_y\d+_class\d+$",
    },
    "colon_cancer_csv": {
        # 标签在顶层 train.csv，**实测列名为 `Id,Type`**（不是文档里暗示的 image_id/label）。
        # 图像位于 train/train/<Id>.png（2190/2190 命中，Id 即文件名 stem）。
        # Type ∈ {Cancer, Connective, Immune, Normal}；按本项目统一约定：
        # Cancer → 1，其余三类归 0（与 lung_colon 的"折叠"一致）。
        "root": MS / "colon-cancer-datasets",
        "label_from": "csv",
        "csv_name": "train.csv",
        "csv_image_col": "Id",
        "csv_label_col": "Type",
        "csv_image_subdir": "train/train",
        "csv_image_ext": ".png",
        "raw_label_map": {"Cancer": 1, "Connective": 0, "Immune": 0, "Normal": 0},
        "organ": "colon",
        "note": "结肠：腺癌 vs 结缔组织/免疫细胞/正常（CSV 标签，折叠；恶性仅占 7.2%）",
    },
    "pcam_lymph": {
        # PatchCamelyon（Camelyon16 派生）：淋巴结转移，**二元标注且完全平衡**。
        # 图像在 HDF5（262,144 × 96×96×3 uint8），标签与分组信息在 meta CSV：
        #   center_tumor_patch  True/False 恰好 131,072 / 131,072（PCam 官方标签定义）
        #   wsi                 216 个切片标识 —— 用于**切片级**分组（比图像级更严格）
        # 这是唯一采用 HDF5 储存的来源，因此 label_from = "h5csv"，
        # 由 `list_items` 直接从 h5 读取张量、按行号与 CSV 对齐。
        "root": MS / "pcam" / "histopathologic-cancer-detection",
        "label_from": "h5csv",
        "h5_name": "camelyonpatch_level_2_split_train_x.h5",
        "h5_dataset": "x",
        "csv_name": "camelyonpatch_level_2_split_train_meta.csv",
        "csv_label_col": "center_tumor_patch",
        "csv_group_col": "wsi",
        "raw_label_map": {"True": 1, "False": 0},
        "organ": "lymph node",
        "note": "淋巴结转移：p=0.5µm 级 patch，中心为肿瘤 vs 正常（131072/131072，按 wsi 分组）",
    },
    "gastric_tum": {
        # GCHTID（HMU-GC-HE-30K）：8 个组织类型各 3,887 张（类别间平衡）。
        # 映射：恶性 = {TUM}（3,887）；良性 = {NOR, MUS}（2 × 3,887 = 7,774）。
        # 注意：取两类为良性后，该来源**并非 1:1 平衡**（恶性:良性 = 1:2），
        # 评估时按每类上限抽样，故矩阵中使用的是平衡子集（见 Methods 的抽样口径）。
        # 刻意**排除** STR（cancer-associated stroma）：癌相关但非恶性组织，纳入会引入标签噪声；
        # 同理排除 ADI/DEB/MUC/LYM（脂肪/组织碎片/黏液/淋巴细胞），既非肿瘤亦非正常黏膜。
        # 这是**新器官（胃）**；标签体系为 Kather 组织学类别，与 lung_colon 同源但来源队列不同，
        # 因此 lung_colon↔gastric_tum 的跨来源 AUC 主要反映来源差异而非范畴冲突——须在论文写明。
        "root": MS / "gastric_cancer" / "HMU-GC-HE-30K" / "all_image",
        "malignant": {"TUM"}, "benign": {"NOR", "MUS"},
        "organ": "stomach",
        "note": "胃：肿瘤上皮 vs 正常黏膜+平滑肌（每类 3887 张，取 2 类为良性 → 1:2）",
    },
}


def image_files(d: Path) -> list[Path]:
    return [p for p in sorted(d.glob("*")) if p.suffix.lower() in IMAGE_EXTS]


def list_items(spec: dict) -> list[tuple[Path, int, str]]:
    """返回 [(图像路径, 标签, 分组键)]。

    分组键用于 GroupKFold：有患者 ID 的来源用患者 ID，其余用图像文件名 stem。
    """
    root: Path = spec["root"]
    label_from = spec.get("label_from", "dir")
    mode = spec.get("group_from", "filename")
    rows: list[tuple[Path, int, str]] = []

    if label_from == "dir":
        for d in root.rglob("*"):
            if not d.is_dir():
                continue
            if d.name in spec["malignant"]:
                lab = 1
            elif d.name in spec["benign"]:
                lab = 0
            else:
                continue
            for p in image_files(d):
                grp = d.parent.name if mode == "parent_dir" else p.stem
                rows.append((p, lab, grp))

    elif label_from == "filename":
        pat = re.compile(spec["label_regex"])
        lm: dict[str, int] = spec["filename_labels"]
        gpat = re.compile(spec["group_regex"]) if spec.get("group_regex") else None
        for p in sorted(root.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
                continue
            m = pat.search(p.name)
            if not m:
                continue
            key = m.group(1)
            if key not in lm:
                continue
            # 分组：优先用文件名里的患者 ID（breast_idc），否则按声明的来源取
            if gpat is not None:
                gm = gpat.match(p.stem)
                grp = gm.group(1) if gm else p.stem
            elif mode == "parent_dir":
                grp = p.parent.name
            else:
                grp = p.stem
            rows.append((p, lm[key], grp))

    elif label_from == "csv":
        csv_path = root / spec["csv_name"]
        if not csv_path.exists():
            return []
        sub = root / spec.get("csv_image_subdir", "")
        ext = spec.get("csv_image_ext", "")
        lmap: dict[str, int] = spec["raw_label_map"]
        with open(csv_path, encoding="utf-8-sig", newline="") as fh:
            for r in csv.DictReader(fh):
                fname = (r.get(spec["csv_image_col"]) or "").strip()
                raw = (r.get(spec["csv_label_col"]) or "").strip()
                if not fname or raw not in lmap:
                    continue
                # CSV 可能只给 ID（不含扩展名），此时补上该来源声明的扩展名
                if ext and not fname.lower().endswith(ext):
                    fname = fname + ext
                p = (sub / fname) if sub else (root / fname)
                if not p.exists():
                    hits = list(root.rglob(Path(fname).name))
                    if not hits:
                        continue
                    p = hits[0]
                rows.append((p, lmap[raw], p.stem))

    elif label_from == "h5csv":
        # 图像在 HDF5、标签与分组在 CSV（PatchCamelyon）。
        # 这里为**每一行**返回一个"合成路径" `h5path::<row_index>`，
        # 供 60_cross_source_matrix.py 的 cmd_extract 按行号从 h5 取图；
        # 分组键取自 CSV 的 wsi 列（切片级），比图像级更严格。
        import csv as _csv

        import h5py
        h5_path = root / spec["h5_name"]
        csv_path = root / spec["csv_name"]
        if not h5_path.exists() or not csv_path.exists():
            return []
        lmap = spec["raw_label_map"]
        with h5py.File(h5_path, "r") as h:
            n_img = int(h[spec["h5_dataset"]].shape[0])
        with open(csv_path, encoding="utf-8-sig", newline="") as fh:
            for i, r in enumerate(_csv.DictReader(fh)):
                if i >= n_img:
                    break
                raw = (r.get(spec["csv_label_col"]) or "").strip()
                if raw not in lmap:
                    continue
                grp = (r.get(spec.get("csv_group_col", "")) or f"row{i}").strip()
                rows.append((Path(f"{h5_path}::{i}"), lmap[raw], grp))

    else:
        raise ValueError(f"未知 label_from={label_from!r}（来源 {spec.get('organ')!r}）")

    return rows


def discover(only: list[str] | None = None) -> dict[str, dict]:
    """只保留实际存在且两类都非空的数据集，并附计数。"""
    out: dict[str, dict] = {}
    for name, spec in SPECS.items():
        if only and name not in only:
            continue
        if not spec["root"].exists():
            continue
        rows = list_items(spec)
        if not rows:
            continue
        n_pos = sum(1 for _, l, _ in rows if l == 1)
        n_neg = sum(1 for _, l, _ in rows if l == 0)
        if n_pos and n_neg:
            out[name] = {**spec, "rows": rows, "n_pos": n_pos, "n_neg": n_neg,
                         "n_total": len(rows)}
    return out


def _path_index() -> dict[str, tuple[str, str]]:
    """path(str) → (来源名, 分组键)。首次调用时构建，之后缓存。"""
    global _INDEX
    if _INDEX is None:
        idx: dict[str, tuple[str, str]] = {}
        for name, spec in SPECS.items():
            for p, _lab, grp in list_items(spec):
                idx[str(p)] = (name, grp)
        _INDEX = idx
    return _INDEX


_INDEX: dict[str, tuple[str, str]] | None = None


_H5_ROW_GROUP: dict[str, list[str]] | None = None


def _h5_row_groups() -> dict[str, list[str]]:
    """h5 文件路径 → 每一行的分组键（wsi）。

    PCam 的特征 npz 里存的是 `*.h5::<row>` 合成路径，反查注册表索引时
    会在 `::` 处断掉、退化成整数 stem。因此对 h5 来源单独按**行号**取分组键。
    """
    global _H5_ROW_GROUP
    if _H5_ROW_GROUP is None:
        import csv as _csv

        out: dict[str, list[str]] = {}
        for spec in SPECS.values():
            if spec.get("label_from") != "h5csv":
                continue
            csv_path = spec["root"] / spec["csv_name"]
            if not csv_path.exists():
                continue
            vals: list[str] = []
            with open(csv_path, encoding="utf-8-sig", newline="") as fh:
                for r in _csv.DictReader(fh):
                    vals.append((r.get(spec.get("csv_group_col", "")) or "").strip())
            out[str(spec["root"] / spec["h5_name"])] = vals
        _H5_ROW_GROUP = out
    return _H5_ROW_GROUP


def group_key_for(path: str, fallback_name: str | None = None) -> str:
    """给定图像路径，返回该来源定义的**评估分组键**。

    已有 npz 特征里只存了 paths，没有存分组。为了不重提全部特征，
    这里按路径反查注册表拿分组键；查不到时退回文件 stem（历史行为）。

    对带患者 ID 的来源（`breast_idc`）这会返回患者 ID，从而把评估从
    "按图像编号分组"升级为"按患者分组"——这正是该来源的独特价值。
    """
    sp = str(path)
    # h5 来源：`<h5path>::<row>` —— 按行号取 wsi（切片级分组）
    if "::" in sp:
        h5f, _, row = sp.rpartition("::")
        vals = _h5_row_groups().get(h5f)
        if vals and row.isdigit() and int(row) < len(vals) and vals[int(row)]:
            return vals[int(row)]
    hit = _path_index().get(sp)
    if hit:
        return hit[1]
    return Path(sp).stem


def groups_for(paths) -> list[str]:
    """批量取分组键（评估时用）。"""
    return [group_key_for(p) for p in paths]


def summary_rows(specs: dict[str, dict]) -> list[dict]:
    """给表格/打印用的紧凑摘要。"""
    out = []
    for name, s in specs.items():
        out.append({"source": name, "organ": s["organ"], "n_pos": s["n_pos"],
                    "n_neg": s["n_neg"], "n_total": s["n_total"],
                    "label_from": s.get("label_from", "dir"),
                    "note": s.get("note", "")})
    return out


def audit_counts(specs: dict[str, dict]) -> str:
    lines = [f"{'来源':16s} {'器官':12s} {'恶性':>7s} {'良性':>7s} {'合计':>7s}"]
    for r in summary_rows(specs):
        lines.append(f"{r['source']:16s} {r['organ']:12s} {r['n_pos']:7d} "
                     f"{r['n_neg']:7d} {r['n_total']:7d}")
    return "\n".join(lines)
