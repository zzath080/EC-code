"""生成仓库整理报告（UTF-8）。

为什么要单独一个脚本，而不是 PowerShell 重定向
----------------------------------------------
`python scripts/96_organize_repo.py --audit > docs/Repo_Cleanup_Report.md` 在
PowerShell 里会写出 **UTF-16**，首字节 0xFF，任何 UTF-8 读取都会失败
（本项目已因编码问题损失过一次文档，见 README 的编码警告）。
因此报告一律用 Python 以 `encoding="utf-8"` 写入，并立刻回读校验。

另外：审计输出有相当一部分走 **stderr**，只捕 stdout 会得到一个残缺报告
（实测只拿到 1255 字符）。两路都要收。

用法：
    python scripts/99_write_cleanup_report.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "Repo_Cleanup_Report.md"

# 用原始字符串避免 \a 之类被当作转义（曾把 `archive/` 变成响铃符）
HEADER = r"""# 仓库整理报告（由 `scripts/96_organize_repo.py` 生成）

> 生成时间：2026-10-12。可用 `python scripts/96_organize_repo.py --audit` 随时复现。
> 已执行清理：释放 **30.6 GB**；清理前 137.2 GB，清理后 106.6 GB
> （再计入新增的权重/特征资产包 2.4 GB，当前仓库 109.0 GB）。
> 分类规则：**只删「可证明可重建」的项**；原始数据与中间产物一律不自动删。

## 不可删的两个项（容易被误判为垃圾）

1. **`.pylibs/`（4.6 GB）** —— 它不是缓存，而是本项目唯一的 torch / transformers 来源；
   Anaconda 基环境里这两个包**不存在**（已实测）。删掉即无法复现特征提取与 WSI 训练。
2. **`archive/`（4.0 GB）** —— "为什么放弃某些设计"的唯一记录，删掉就永久失去依据。
3. **`data/raw/tcga_ucec_tumor_subset/` 的 3 个 `.svs.part`（2.8 GB）** ——
   `01_gdc_download.py` 支持 Range 续传，删除等于丢弃已下载进度。

## 审计输出

```
"""


def main() -> int:
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "96_organize_repo.py"), "--audit"],
                       capture_output=True, cwd=str(ROOT))
    # Windows 下子进程 stdout 是 \r\n；若再以文本模式写入，Python 会把 \n 转成 \r\n，
    # 于是 \r\n 变成 \r\r\n，渲染出来是**双倍行距**（实测每行后多一个空行）。
    # 先统一成 \n，并用 newline="\n" 明确禁止二次转换。
    body = (r.stdout.decode("utf-8", errors="replace")
            + r.stderr.decode("utf-8", errors="replace")).replace("\r\n", "\n").replace("\r", "\n")
    if not body.strip():
        print("[ERROR] 审计无输出，报告未生成", file=sys.stderr)
        return 1
    text = HEADER + body.rstrip() + "\n```\n"
    OUT.write_text(text, encoding="utf-8", newline="\n")
    back = OUT.read_text(encoding="utf-8")
    assert back.startswith("# 仓库整理报告"), "回读校验失败"
    assert "archive/" in back, "转义问题：archive/ 未正确写入"
    n_blank = sum(1 for x in back.splitlines() if not x.strip())
    print(f"报告 -> {OUT.relative_to(ROOT)}  {len(back)} 字符，"
          f"{len(back.splitlines())} 行（空行 {n_blank}），UTF-8 回读校验通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
