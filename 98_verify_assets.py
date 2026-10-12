"""校验资产包：zip 内容与 asset_manifest.json 的 SHA256 必须逐项一致。

为什么需要
----------
整理仓库时会移动/删除文件。若打包过程有误（漏文件、截断、写错路径），
"看起来打好了"的包会在别人手里才暴露问题。本脚本做**端到端校验**：

1. 打开两个资产 zip，逐条目计算 SHA256；
2. 与 `data/manifests/asset_manifest.json` 中对应文件的 sha256 比对；
3. 检查"清单里有但包里没有"与"包里有但清单没有"两个方向；
4. 顺带校验 `dist/assets/CHECKSUMS.sha256` 里记录的 zip 自身哈希。

用法：
    python scripts/98_verify_assets.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifests" / "asset_manifest.json"
ASSETS = ROOT / "dist" / "assets"


def sha256_file(p: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def in_zip_hashes(zp: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with zipfile.ZipFile(zp) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f"{zp.name} 内条目损坏：{bad}")
        for info in zf.infolist():
            if info.is_dir():
                continue
            h = hashlib.sha256()
            with zf.open(info) as fh:
                while True:
                    b = fh.read(1 << 22)
                    if not b:
                        break
                    h.update(b)
            out[info.filename] = h.hexdigest()
    return out


def main() -> int:
    if not MANIFEST.exists():
        print(f"[ERROR] 缺 {MANIFEST}", file=sys.stderr)
        return 2
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    ok = bad = 0
    problems: list[str] = []

    checks = [
        ("BSPC_weights.zip", {k: v["sha256"] for k, v in m["weights"].items()}),
        ("BSPC_features.zip",
         {k.replace("data/features/", "features/"): v["sha256"]
          for k, v in m["features"].items()}),
    ]
    for name, expect in checks:
        zp = ASSETS / name
        if not zp.exists():
            problems.append(f"缺资产包 {name}")
            continue
        got = in_zip_hashes(zp)
        print(f"--- {name}：包内 {len(got)} 项，清单 {len(expect)} 项 ---")
        for k, want in sorted(expect.items()):
            if k not in got:
                problems.append(f"{name}: 清单有但包内缺 -> {k}")
                bad += 1
            elif got[k] != want:
                problems.append(f"{name}: SHA256 不符 -> {k}")
                bad += 1
            else:
                ok += 1
        for k in sorted(set(got) - set(expect)):
            problems.append(f"{name}: 包内有但清单无 -> {k}")
            bad += 1

    # CHECKSUMS.sha256 校验 zip 自身
    ck = ASSETS / "CHECKSUMS.sha256"
    if ck.exists():
        print("--- CHECKSUMS.sha256 ---")
        for line in ck.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            h, fname = line.split(None, 1)
            fname = fname.strip()
            p = ASSETS / fname
            if not p.exists():
                problems.append(f"CHECKSUMS 指向不存在的包 {fname}")
                bad += 1
                continue
            actual = sha256_file(p)
            if actual == h:
                print(f"  [OK]   {fname}")
                ok += 1
            else:
                problems.append(f"CHECKSUMS 不符 {fname}")
                bad += 1

    print()
    print(f"[资产校验] 通过 {ok} 项，问题 {bad} 项")
    if problems:
        for p in problems:
            print("   -", p)
        return 1
    print("全部与 asset_manifest.json 一致 ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
