#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
53_extract_rar.py
=================
自动探测可用工具并解压 RAR（含 RAR5）。

背景（实测）：
    ENDO-AID 的 .rar 文件头为 `Rar!\\x1a\\x07\\x01\\x00` → **RAR5**。
    本机 libarchive 3.6.2 / bsdtar 3.6.2 **不支持 RAR5**（需 ≥ 3.7）。
    可行的替代：7-Zip（原生支持 RAR5），可用 winget 安装：
        winget install --id 7zip.7zip --accept-package-agreements --accept-source-agreements

本脚本按优先级自动选择：7z → 7za → unrar → bsdtar → libarchive，
并给出明确的失败原因与下一步建议。

用法：
    python scripts/53_extract_rar.py --rar data/raw/endoaid_external/xxx.rar --list
    python scripts/53_extract_rar.py --rar ... --dest data/raw/endoaid_external/extracted
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CANDIDATE_7Z = [
    "7z", "7za", "7zr",
    r"C:\Program Files\7-Zip\7z.exe",
    r"C:\Program Files (x86)\7-Zip\7z.exe",
]


def find_7z() -> str | None:
    for c in CANDIDATE_7Z:
        p = shutil.which(c) or (c if Path(c).exists() else None)
        if p:
            return p
    return None


def rar_version(path: Path) -> str:
    with open(path, "rb") as fh:
        head = fh.read(8)
    if head.startswith(b"Rar!\x1a\x07\x01"):
        return "RAR5"
    if head.startswith(b"Rar!\x1a\x07\x00"):
        return "RAR4"
    return f"未知 ({head[:8]!r})"


def run(cmd: list[str], timeout: int = 3600) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except Exception as exc:  # noqa: BLE001
        return -1, f"{type(exc).__name__}: {exc}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rar", required=True)
    ap.add_argument("--dest", default=None)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    rar = Path(args.rar)
    if not rar.is_absolute():
        rar = ROOT / rar
    if not rar.exists():
        print(f"[ERROR] 文件不存在：{rar}", file=sys.stderr)
        return 2

    ver = rar_version(rar)
    print(f"文件：{rar.name}  {rar.stat().st_size/1024**3:.2f} GB")
    print(f"格式：{ver}")
    dest = Path(args.dest) if args.dest else rar.parent / "extracted"
    if not dest.is_absolute():
        dest = ROOT / dest
    dest.mkdir(parents=True, exist_ok=True)

    # 1) 7-Zip（唯一能可靠处理 RAR5 的现成工具）
    seven = find_7z()
    if seven:
        print(f"\n使用 7-Zip：{seven}")
        mode = ["l"] if args.list else ["x", f"-o{dest}", "-y"]
        rc, out = run([seven] + mode + [str(rar)], timeout=1800)
        print(out[-3000:] if out else "")
        if rc == 0:
            print(f"\n{'列出成功' if args.list else '解压完成'} -> {dest}")
            return 0
        print(f"[WARN] 7-Zip 返回 {rc}")

    # 2) unrar / unar
    for cmd, argv in (("unrar", ["unrar", "lt" if args.list else "x", str(rar)]),
                      ("unar", ["unar", "-o", str(dest), str(rar)])):
        exe = shutil.which(cmd)
        if exe:
            print(f"\n使用 {cmd}")
            rc, out = run(argv)
            print(out[-2000:])
            if rc == 0:
                return 0

    # 3) bsdtar / libarchive（本机实测不支持 RAR5）
    bsdtar = shutil.which("bsdtar")
    if bsdtar:
        print(f"\n尝试 bsdtar（注意：本机 3.6.2 不支持 RAR5，预计失败）")
        rc, out = run([bsdtar, "-tf" if args.list else "-xf", str(rar)] +
                      ([] if args.list else ["-C", str(dest)]))
        print(out[:800])
        if rc == 0:
            print("bsdtar 成功（说明该文件实际是 RAR4 或 libarchive 支持）")
            return 0
        print("[INFO] bsdtar 失败，符合预期（RAR5 需要 7-Zip）")

    print("\n" + "=" * 70)
    print("无法解压：本机没有支持 RAR5 的工具。请任选一种方式：")
    print("  A) 用 winget 安装 7-Zip（推荐）：")
    print("     winget install --id 7zip.7zip --accept-package-agreements --accept-source-agreements")
    print("     安装后脚本会自动找到 C:\\Program Files\\7-Zip\\7z.exe")
    print("  B) 手动下载 7-Zip 官方安装包并安装")
    print("  C) 用较新的 libarchive（≥3.7）替代")
    print("安装完成后重跑本脚本即可。")
    print("=" * 70)
    return 1


if __name__ == "__main__":
    sys.exit(main())
