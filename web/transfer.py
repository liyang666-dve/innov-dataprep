"""rsync 交付模块：dry-run 预览 → 正式传输 → 留痕（.dataprep_ops.jsonl kind=deliver）。

设计（见 web_ui_plan.md §3）：
  - 源：由 Web 页面限定为已扫到的 v2.1/v3.0 数据集（也可手动填绝对路径）；
  - 目标：交付页每次手动填写 host/user/dir（+ 可选 id_rsa），不落盘到 config；
  - 安全：rsync 不在本层做文件过滤，源需是"已软排除好的最终数据集"；
  - 留痕：传输成功后往 ops_file 追加一行 kind=deliver（含目标主机/路径）。
只读/追加操作，不修改源数据。
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from run import load_config, state_path, ops_file  # noqa: E402

RSYNC: str | None = shutil.which("rsync")


def _ssh_cmd(identity: str | None) -> str:
    """-e 的 ssh 命令串。默认禁 StrictHostKeyChecking 免首次确认；可带 -i <id_rsa>。"""
    if identity:
        return f"ssh -o StrictHostKeyChecking=no -o BatchMode=yes -i {identity}"
    return "ssh -o StrictHostKeyChecking=no -o BatchMode=yes"


def _target(host: str, user: str, dir: str) -> str | None:
    host = (host or "").strip()
    dir = (dir or "").strip().rstrip("/")
    if not host or not dir:
        return None
    if ":" in host:  # 已含端口 host:port 或完整 target 的简单兼容
        base = host
    else:
        base = f"{user}@{host}" if user.strip() else host
    return f"{base}:{dir}"


def _rsync_base_args(source: Path, target: str, identity: str | None) -> list[str]:
    return [RSYNC, "-avz", "-e", _ssh_cmd(identity), str(source), target]


# ---------------------------------------------------------------- 预览
def preview(source: str, target: str, identity: str | None = None) -> dict:
    """rsync -avn --dry-run --stats，返回解析后的 {ok, dirname, files, size, error, lines}。"""
    if not RSYNC:
        return {"ok": False, "error": "未安装 rsync（sudo apt install rsync）"}
    src = Path(source).expanduser()
    if not src.is_dir():
        return {"ok": False, "error": f"源目录不存在: {source}"}
    if not target:
        return {"ok": False, "error": "目标 host/user/dir 不完整"}
    cmd = _rsync_base_args(src, target, identity)
    # --dry-run + --stats：只枚举将传输的文件与体积，不出错导致误报
    cmd.insert(1, "-n")
    cmd.insert(2, "--stats")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "预览超时（网络不通？）", "lines": ""}
    out = (r.stdout or "") + (r.stderr or "")
    dirname = src.name
    files, size = None, None
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("Number of regular files transferred:"):
            files = line.split(":", 1)[1].strip()
        elif line.startswith("Total transferred file size:"):
            size = line.split(":", 1)[1].strip()
    return {
        "ok": r.returncode == 0,
        "dirname": dirname,
        "files": files,
        "size": size,
        "error": "" if r.returncode == 0 else (out.strip().splitlines() or ["rsync 预览失败"])[:3],
        "lines": out.strip(),
    }


# ---------------------------------------------------------------- 传输
def do_transfer(source: str, target: str, identity: str | None = None, emit=None):
    """正式传输并流式推日志；成功后追加 deliver 留痕。emit(callable)->逐行推送。"""
    say = emit or (lambda s: print(s))
    if not RSYNC:
        say("[ERROR] 未安装 rsync（sudo apt install rsync）")
        return 1
    src = Path(source).expanduser()
    if not src.is_dir():
        say(f"[ERROR] 源目录不存在: {source}")
        return 1
    cmd = _rsync_base_args(src, target, identity)
    say(f"==> {' '.join(cmd)}")
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1)
    except OSError as e:
        say(f"[ERROR] 无法启动 rsync：{e}")
        return 1
    if p.stdout:
        for line in p.stdout:
            say(line.rstrip("\n"))
    rc = p.wait()
    if rc == 0:
        say(f"[OK] 交付完成 -> {target}/{src.name}")
        _record_deliver(src, target)
    else:
        say(f"[ERROR] rsync 退出码 {rc}")
    say("[exit]")
    return rc


def _record_deliver(src: Path, target: str) -> None:
    try:
        cfg = load_config()
        sf = state_path(cfg, None)
        p = ops_file(sf)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "time": datetime.now().isoformat(timespec="seconds"),
                "action": "deliver",
                "name": src.name,
                "path": str(src),
                "kind": "deliver",
                "note": f"rsync → {target}/{src.name}",
            }, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001
        pass