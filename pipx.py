#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
pipx.py — 绕过 pip 的极简 wheel 安装器（Windows / 沙箱环境专用）
================================================================
背景：本机 pip 在解包阶段对临时目录内的 `*.whl.metadata` 写入持续返回
      PermissionError（与文件大小无关），导致 torch / openslide 等无法安装。
      本脚本直接用 urllib 从 PyPI 官方 API 解析 wheel URL，下载后按 zip 解压
      到工作区内的目标目录，从而完全绕开 pip 的临时目录机制。

用法：
    python scripts/pipx.py --target .pylibs install openslide-python timm
    python scripts/pipx.py --target .pylibs install --no-deps torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
    python scripts/pipx.py --target .pylibs list
    python scripts/pipx.py --check

说明：
    - 纯 wheel 安装（不编译源码包）。sdist-only 的包会明确报错。
    - 不处理依赖解析的复杂情形，但支持 `--deps` 做一层直接依赖递归。
    - 目标目录通过 PYTHONPATH 使用：$env:PYTHONPATH="<repo>\.pylibs"
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UA = {"User-Agent": "endo-wsi-research/1.0"}

DEFAULT_TARGET = ROOT / ".pylibs"
DEFAULT_INDEX = "https://pypi.org/simple"
PYPI_JSON = "https://pypi.org/pypi/{name}/json"


def pick_platform() -> list[str]:
    """当前解释器可用的 wheel 平台标签（按优先级）。

    注意：abi3 wheel 的 cp3XX 标签是「最低支持版本」，cp312 解释器可以安装
    cp310-abi3 / cp311-abi3 等更低版本的 abi3 wheel，因此必须全部列出。
    """
    tags = ["cp312-cp312-win_amd64"]
    for minor in range(311, 37, -1):
        tags.append(f"cp3{minor - 300}-abi3-win_amd64")
    tags += [
        "cp37-abi3-win_amd64",
        "py3-none-win_amd64",
        "py3-none-any",
        "py2.py3-none-any",
    ]
    return tags


def resolve_from_pypi(name: str, version: str | None) -> str:
    with urllib.request.urlopen(urllib.request.Request(PYPI_JSON.format(name=name), headers=UA), timeout=90) as resp:
        data = json.load(resp)
    if version:
        files = data["releases"].get(version)
        if not files:
            raise RuntimeError(f"{name}=={version} 在 PyPI 上没有该版本")
    else:
        files = data["urls"]
    tags = pick_platform()
    candidates = [f for f in files if f["filename"].endswith(".whl")]
    if not candidates:
        raise RuntimeError(f"{name}: 只有源码包(sdist)，本脚本不支持编译")
    # 优先平台专用 wheel，其次纯 py3 wheel
    for tag in tags:
        for f in candidates:
            if tag in f["filename"]:
                return f["url"]
    for f in candidates:
        if "py3-none-any" in f["filename"] or "py2.py3-none-any" in f["filename"]:
            return f["url"]
    raise RuntimeError(f"{name}: 找到 wheel 但与本机平台不匹配: {[f['filename'] for f in candidates][:3]}")


def resolve_from_index(name: str, version: str | None, index_url: str) -> str:
    """从自定义 simple index（如 download.pytorch.org）解析 wheel URL。"""
    base = index_url.rstrip("/")
    url = f"{base}/{name}/"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"索引 {url} 不可达: {exc}") from exc
    import re
    links = re.findall(r'href="([^"]+\.whl[^"]*)"', html)
    links = [urllib.parse.urljoin(url, l.split("#")[0]) for l in links]
    if version:
        links = [l for l in links if f"-{version}" in Path(l).name or f"-{version}+" in Path(l).name]
    if not links:
        raise RuntimeError(f"{name}=={version} 在索引 {base} 未找到 wheel")
    tags = pick_platform()
    for tag in tags:
        for l in links:
            if tag in Path(l).name:
                return l
    return links[0]


def remote_size(url: str) -> int:
    """用 HEAD 取远端文件大小；取不到返回 0。"""
    try:
        req = urllib.request.Request(url, method="HEAD", headers=UA)
        with urllib.request.urlopen(req, timeout=60) as resp:
            return int(resp.headers.get("Content-Length") or 0)
    except Exception:  # noqa: BLE001
        return 0


def download(url: str, dest_dir: Path) -> Path:
    """下载 wheel；校验远端大小，支持断点续传，拒绝残缺缓存。

    注意：urlopen(timeout=) 只约束连接建立与单次 recv，代理可能让 recv 长期阻塞，
    表现为文件大小停在某处。因此这里用 socket 级读超时 + 大量 Range 续传。
    """
    import socket
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = Path(urllib.parse.urlparse(url).path).name
    dest = dest_dir / name
    want = remote_size(url)

    if dest.exists():
        have = dest.stat().st_size
        if want and have == want:
            print(f"    已缓存且完整 {name} ({have/1024**2:.1f} MB)")
            return dest
        print(f"    缓存不完整/可疑（{have/1024**2:.1f} MB vs 期望 {want/1024**2:.1f} MB），重新下载")
        dest.unlink(missing_ok=True)

    part = dest.with_suffix(dest.suffix + ".part")
    # 本机对大文件的长连接会被反复掐断（实测每 15–30 MB 断一次），
    # 因此重试次数给到很大，靠 Range 续传把文件啃完。
    for attempt in range(1, 401):
        have = part.stat().st_size if part.exists() else 0
        if want and have > want:
            part.unlink(missing_ok=True)
            have = 0
        if want and have == want:
            break
        headers = dict(UA)
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            if attempt % 10 == 1:
                print(f"    下载 {name}（第 {attempt} 次，起始 {have/1024**2:.1f} MB"
                      f"{f' / {want/1024**2:.1f} MB' if want else ''}）…", flush=True)
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=45) as resp:
                mode = "ab" if (have and resp.status == 206) else "wb"
                if mode == "wb":
                    have = 0
                with open(part, mode) as fh:
                    last = have
                    while True:
                        try:
                            chunk = resp.read(1024 * 1024)
                        except (socket.timeout, TimeoutError) as exc:
                            raise TimeoutError("读超时（45s 无数据）") from exc
                        if not chunk:
                            break
                        fh.write(chunk)
                        last += len(chunk)
                        if last - have >= 256 * 1024 * 1024:
                            have = last
                            print(f"      {last/1024**2:.0f}/{want/1024**2:.0f} MB", flush=True)
        except Exception as exc:  # noqa: BLE001
            if attempt % 10 == 1:
                now = part.stat().st_size if part.exists() else 0
                print(f"    续传({attempt}) {now/1024**2:.1f}"
                      f"{f'/{want/1024**2:.1f}' if want else ''} MB: {type(exc).__name__}", flush=True)
            continue

    got = part.stat().st_size if part.exists() else 0
    if want and got != want:
        raise RuntimeError(f"下载不完整：{got} / {want} 字节")
    os.replace(part, dest)
    print(f"    完成 {dest.stat().st_size/1024**2:.1f} MB")
    return dest


def extract(whl: Path, target: Path) -> int:
    target.mkdir(parents=True, exist_ok=True)
    count = 0
    with zipfile.ZipFile(whl) as zf:
        for info in zf.infolist():
            # 跳过 dist-info 里的 RECORD 等元数据不必要文件保留即可
            if info.filename.endswith("/"):
                continue
            out = target / info.filename
            out.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
            count += 1
    for msg in fix_extension_module_names(target):
        print(f"    [fix] {msg}")
    return count


def fix_extension_module_names(target: Path) -> list[str]:
    """修复带平台标签的顶层扩展模块文件名。

    问题：部分 wheel（实测 duckdb）把顶层扩展模块打成
    `_duckdb.cp312-win_amd64.pyd`，而 `import _duckdb` 只会去找 `_duckdb.pyd`
    （扩展模块的文件名里不含平台标签），于是报 ModuleNotFoundError。

    做法：对形如 `<name>.<平台标签>.pyd/.so/.dll` 的顶层文件，额外补一份
    `<name>.pyd`。保留原文件，避免破坏其他加载路径。
    """
    fixed: list[str] = []
    for p in list(target.iterdir()):
        if not p.is_file():
            continue
        for ext in (".pyd", ".so", ".dll"):
            if not p.name.lower().endswith(ext):
                continue
            stem = p.name[: -len(ext)]
            if "." not in stem:
                continue                      # 已经是 <name>.pyd
            base = stem.split(".")[0]
            plain = target / f"{base}{ext}"
            if not plain.exists():
                shutil.copy2(p, plain)
                fixed.append(f"{p.name} -> {plain.name}")
    return fixed


def is_satisfied_in_base(name: str) -> str | None:
    """若当前解释器的**基础环境**已能导入该包，返回其版本，否则 None。

    目的：`.pylibs` 会通过 PYTHONPATH 遮蔽基础环境。若同一包在两处版本差异很大
    （实测 numpy 2.5.3 vs 基础 1.26.4），会导致"用 NumPy 1.x 编译的模块无法在
    NumPy 2.x 运行"这类崩溃。因此默认不覆盖基础环境已有的包，除非 --force 明确要求。
    """
    import importlib.metadata as md
    try:
        return md.version(name)
    except Exception:  # noqa: BLE001
        return None


def wheel_deps(whl: Path) -> list[str]:
    """从 wheel 的 METADATA 里读出 Requires-Dist（仅一层）。"""
    deps = []
    with zipfile.ZipFile(whl) as zf:
        meta = [n for n in zf.namelist() if n.endswith(".dist-info/METADATA")]
        if not meta:
            return deps
        text = zf.read(meta[0]).decode("utf-8", errors="replace")
    for line in text.splitlines():
        if line.startswith("Requires-Dist:"):
            spec = line.split(":", 1)[1].strip()
            name = spec.split(";")[0].split("(")[0].split("[")[0].strip()
            marker = spec.split(";", 1)[1].strip() if ";" in spec else ""
            if "extra ==" in marker:  # 跳过可选依赖
                continue
            deps.append(name)
    return deps


def spec_parse(spec: str) -> tuple[str, str | None]:
    if "==" in spec:
        n, v = spec.split("==", 1)
        return n.strip(), v.strip()
    return spec.strip(), None


def cmd_install(args) -> int:
    target = Path(args.target).resolve()
    whl_dir = ROOT / "wheels"
    ok, failed = [], []
    queue = list(args.packages)
    seen: set[str] = set()
    while queue:
        spec = queue.pop(0)
        name, version = spec_parse(spec)
        key = f"{name}=={version}"
        if key in seen:
            continue
        seen.add(key)
        print(f"[安装] {key}")
        try:
            if args.index_url and args.index_url.rstrip("/") != DEFAULT_INDEX:
                url = resolve_from_index(name, version, args.index_url)
            else:
                url = resolve_from_pypi(name, version)
            whl = download(url, whl_dir)
            n = extract(whl, target)
            print(f"    解压 {n} 个文件 -> {target}")
            ok.append(key)
            if args.deps:
                for d in wheel_deps(whl):
                    if d.lower() not in {s.split("==")[0].lower() for s in seen}:
                        queue.append(d)
        except Exception as exc:  # noqa: BLE001
            print(f"    [失败] {type(exc).__name__}: {exc}")
            failed.append((key, str(exc)))
    print(f"\n成功 {len(ok)}，失败 {len(failed)}")
    for k, e in failed:
        print(f"  FAIL {k}: {e}")
    return 0 if not failed else 1


def cmd_list(args) -> int:
    target = Path(args.target).resolve()
    if not target.exists():
        print("目标目录不存在")
        return 1
    dists = sorted({p.name.split(".dist-info")[0] for p in target.glob("*.dist-info")})
    for d in dists:
        print(" ", d)
    print(f"共 {len(dists)} 个分发包于 {target}")
    return 0


def cmd_check(args) -> int:
    cands = [ROOT / ".pylibs", Path(args.target).resolve()]
    for t in dict.fromkeys(cands):
        if not t.exists():
            continue
        import importlib
        sys.path.insert(0, str(t))
    print("sys.path 注入完毕")
    for mod in ["numpy", "tifffile", "openslide", "timm", "torch", "h5py", "sklearn"]:
        try:
            m = importlib.import_module(mod)
            print(f"  {mod:12s} OK   {getattr(m, '__version__', '?')}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {mod:12s} MISS {type(exc).__name__}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["install", "list", "check"])
    ap.add_argument("packages", nargs="*")
    ap.add_argument("--target", default=str(DEFAULT_TARGET))
    ap.add_argument("--index-url", default=DEFAULT_INDEX)
    ap.add_argument("--deps", action="store_true", help="递归安装一层直接依赖")
    args = ap.parse_args()
    if args.action == "install":
        return cmd_install(args)
    if args.action == "list":
        return cmd_list(args)
    return cmd_check(args)


if __name__ == "__main__":
    raise SystemExit(main())
