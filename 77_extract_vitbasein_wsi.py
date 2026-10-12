# 为 vit-base-in（第三编码器）补齐 TCGA WSI 级特征。
#
# 为什么写成 .py 而不是直接在各处敲命令：
#   1. `--wsi-dirs` 是 nargs="*"，在 PowerShell 里传逗号分隔值会被当成单个字符串，
#      find_wsi 会找不到切片而不报错——必须显式传数组；
#   2. 活检源域需要 tcga_normal(53) + tcga_tumor(48) 两个队列，
#      而各队列的 WSI 分散在 3 个目录，需要统一搜索；
#   3. 需要可断点续跑（--skip-existing），中断后不用从头再来。
#
# 用法：
#   python scripts/77_extract_vitbasein_wsi.py            # 全部 4 个队列
#   python scripts/77_extract_vitbasein_wsi.py --cohorts tcga_normal tcga_tumor
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

# WSI 实际分散的目录（与 55_external_biopsy_validation.py 的源域定义保持一致）
WSI_DIRS = [
    "data/raw/tcga_ucec_normals",
    "data/raw/tcga_ucec_tumor_subset",
    "data/raw/tcga_ucec_01z",
    "TCGA-UCEC",
]

# 队列 → patch 索引目录。注意 tcga_tumor 里含 2 张 01Z，这是仓库既有状态，
# 不在这里"修正"——改了会让 vit-base-in 与 phikon/phikon-v2 的特征集合不可比。
COHORTS = {
    "tcga_normal": "data/patches/tcga_normal",
    "tcga_tumor": "data/patches/tcga_tumor",
    "tcga_tumor_legacy": "data/patches/tcga_tumor_legacy",
    "tcga_01z": "data/patches/tcga_01z",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="vit-base-in")
    ap.add_argument("--cohorts", nargs="*", default=list(COHORTS))
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    rc_all = 0
    for cohort in args.cohorts:
        if cohort not in COHORTS:
            print(f"[SKIP] 未知队列 {cohort}")
            continue
        feat_dir = ROOT / "data" / "features" / args.encoder / cohort
        n_have = len(list(feat_dir.glob("*.h5"))) if feat_dir.exists() else 0
        n_need = len(list((ROOT / COHORTS[cohort]).glob("*.csv")))
        print(f"\n=== {args.encoder} / {cohort}：已有 {n_have}，索引 {n_need} ===", flush=True)
        cmd = [
            PY, str(ROOT / "scripts" / "12_extract_features.py"),
            "--patches", COHORTS[cohort],
            "--cohort", cohort,
            "--encoder", args.encoder,
            "--wsi-dirs", *WSI_DIRS,          # 显式展开为多个参数，避免被当成单个字符串
            "--save", "--skip-existing",
            "--batch-size", str(args.batch_size),
        ]
        r = subprocess.run(cmd, cwd=str(ROOT))
        if r.returncode != 0:
            print(f"[WARN] {cohort} 返回 {r.returncode}", flush=True)
            rc_all = r.returncode
    return rc_all


if __name__ == "__main__":
    sys.exit(main())
