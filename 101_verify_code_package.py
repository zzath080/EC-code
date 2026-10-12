"""核对代码备份包：README 自动生成的说明列必须有信息量。

为什么单独写
------------
第一次生成时，"首行说明"列把**文件名本身**抓了进去（97 行全是文件名），
因为那些脚本的 docstring 第一行就是文件名、第二行是 `====` 分隔线。
这个检查把"说明列是否退化"变成可判定的：说明不能等于文件名，也不能为空。

用法：
    python scripts/101_verify_code_package.py
"""
from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"


def main() -> int:
    zips = sorted(DIST.glob("BSPC_code_*.zip"))
    if not zips:
        print("[ERROR] 未找到 dist/BSPC_code_*.zip", file=sys.stderr)
        return 2
    z = zips[-1]
    print(f"检查 {z.name}（{z.stat().st_size/1024:.0f} KB）")

    with zipfile.ZipFile(z) as f:
        names = f.namelist()
        rd = f.read("README_CODE.md").decode("utf-8")
        req = f.read("requirements.txt").decode("utf-8")
        run = f.read("run.ps1").decode("utf-8")

    problems: list[str] = []

    # 1) 说明列不得退化
    row = re.compile(r"^\| `([^`]+)` \| (.*?) \|$")
    rows = [row.match(x) for x in rd.splitlines()]
    rows = [m for m in rows if m]
    degenerate = [m.group(1) for m in rows if m.group(2).strip() in ("", m.group(1))]
    print(f"  脚本表行数 {len(rows)}；说明退化 {len(degenerate)}")
    if degenerate:
        problems.append(f"说明列为空或等于文件名：{degenerate[:6]}")

    # 2) 包内脚本数应与磁盘一致。
    #    例外：**本校验器自身**是在包生成之后才创建的，因此永远比包内多 1 个。
    #    这是时序差异，不是缺文件——要显式识别，否则这条检查会永远"失败"。
    disk = {p.name for p in (ROOT / "scripts").glob("*.py")}
    inzip = {n.split("/")[-1] for n in names
             if n.startswith("scripts/") and n.endswith(".py")}
    only_disk = sorted(disk - inzip)
    only_zip = sorted(inzip - disk)
    self_name = Path(__file__).name
    only_disk_real = [x for x in only_disk if x != self_name]
    print(f"  scripts/ 磁盘 {len(disk)} 个，包内 {len(inzip)} 个"
          f"（仅磁盘有：{only_disk}；仅包内有：{only_zip}）")
    if only_disk_real:
        problems.append(f"磁盘有但包内缺：{only_disk_real}")
    if only_zip:
        problems.append(f"包内有但磁盘缺：{only_zip}")

    # 3) 可移植性：不得硬编码本机路径
    host_marks = ["D:\\Softwares", "C:\\Users\\AndrewChan", "Desktop\\跨域"]
    for label, text in (("README_CODE.md", rd), ("run.ps1", run)):
        for h in host_marks:
            if h in text and label == "run.ps1":
                problems.append(f"{label} 硬编码本机路径：{h}")
    print(f"  run.ps1 硬编码本机路径：{'有' if any(h in run for h in host_marks) else '无'}")

    # 4) requirements 不得把标准库写进去
    stdlib_leak = [m for m in ("os", "sys", "json", "re", "pathlib", "collections",
                               "argparse", "subprocess", "hashlib")
                   if re.search(rf"^{m}$", req, flags=re.M)]
    print(f"  requirements 混入标准库：{stdlib_leak or '无'}")
    if stdlib_leak:
        problems.append(f"requirements 混入标准库：{stdlib_leak}")

    # 5) 必需的关键脚本在不在
    need = ["source_registry.py", "59_stain_and_shortcut_audit.py",
            "60_cross_source_matrix.py", "91_shortcut_residualization.py",
            "92_probe_robustness.py", "93_trivial_baselines.py",
            "env_check.py"]
    missing = [x for x in need if f"scripts/{x}" not in names]
    print(f"  关键脚本缺失：{missing or '无'}")
    if missing:
        problems.append(f"关键脚本缺失：{missing}")

    print()
    if problems:
        print(f"[代码包校验] 问题 {len(problems)} 项：")
        for x in problems:
            print("   -", x)
        return 1
    print("[代码包校验] 全部通过 ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
