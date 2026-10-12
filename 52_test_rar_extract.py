#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
52_test_rar_extract.py
======================
测试本机能否解压 RAR5（ENDO-AID 的 .rar 文件头为 `Rar!\x1a\x07\x01\x00` 即 RAR5）。

libarchive / bsdtar 对 RAR 的支持通常只到 RAR3（或需编译时启用 unrar），
因此必须先实测；若不可用，需要另找方案（否则下载完了也打不开）。

用法：
    python scripts/52_test_rar_extract.py --rar <path>
    python scripts/52_test_rar_extract.py --probe-only   # 只报告可用工具
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / ".pylibs"
if LIB.exists():
    sys.path.insert(0, str(LIB))


def report_tools() -> dict:
    out: dict = {}
    for cmd in ("bsdtar", "tar", "7z", "7za", "unrar", "unar", "lsar"):
        from shutil import which
        p = which(cmd)
        out[cmd] = p
        print(f"  {'✓' if p else '✗'} {cmd:8s} {p or ''}")
    try:
        import libarchive
        out["python:libarchive"] = getattr(libarchive, "__file__", "ok")
        print(f"  ✓ python:libarchive")
    except Exception as exc:  # noqa: BLE001
        out["python:libarchive"] = None
        print(f"  ✗ python:libarchive ({type(exc).__name__})")
    return out


def try_libarchive(rar: Path) -> tuple[bool, str]:
    try:
        import libarchive
    except Exception as exc:  # noqa: BLE001
        return False, f"libarchive 不可用: {exc}"
    try:
        # 只读目录，不解压
        with libarchive.file_reader(str(rar)) as archive:
            names = []
            for i, entry in enumerate(archive):
                names.append(entry.pathname)
                if i >= 5:
                    break
            return True, f"可读，前几个条目: {names}"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def try_bsdtar(rar: Path) -> tuple[bool, str]:
    from shutil import which
    exe = which("bsdtar")
    if not exe:
        return False, "bsdtar 不存在"
    try:
        p = subprocess.run([exe, "-tf", str(rar)], capture_output=True, text=True, timeout=120)
        if p.returncode == 0:
            lines = [l for l in p.stdout.splitlines() if l.strip()][:5]
            return True, f"可读，前几个条目: {lines}"
        return False, f"returncode={p.returncode} stderr={p.stderr[:200]}"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rar", default="data/raw/endoaid_external/endometrium-carcinoma-pipelle.rar")
    ap.add_argument("--probe-only", action="store_true")
    args = ap.parse_args()

    print("可用解压工具：")
    report_tools()
    if args.probe_only:
        return 0

    rar = ROOT / args.rar
    if not rar.exists():
        print(f"\n[INFO] {rar} 尚不存在（可能仍在下载）")
        return 0
    print(f"\n目标文件 {rar.name}  {rar.stat().st_size/1024**3:.2f} GB")
    with open(rar, "rb") as fh:
        head = fh.read(8)
    print(f"文件头 {head!r}  -> {'RAR5' if head.startswith(b'Rar!\x1a\x07\x01') else 'RAR4 或非 RAR'}")

    for name, fn in (("bsdtar", try_bsdtar), ("libarchive", try_libarchive)):
        ok, msg = fn(rar)
        print(f"\n[{name}] {'成功' if ok else '失败'}: {msg}")

    print("\n结论：若两者都失败，需要以下任一方案——")
    print("  1) 安装 7-Zip（含 RAR5 支持）后调用 7z x")
    print("  2) winget install 7zip.7zip   （或用官方安装包）")
    print("  3) 找该数据集的其他分发格式（若有 TIFF 直链）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
