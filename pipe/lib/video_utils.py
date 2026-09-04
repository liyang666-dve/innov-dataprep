"""视频工具：用系统 ffprobe 读帧数 / 分辨率（免 PyAV / torch 依赖）。

策略：先读容器头里的 nb_frames（秒回）；拿不到再 -count_frames 全量解码。
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

FFPROBE: str | None = shutil.which("ffprobe") or shutil.which("ffprobe.exe")
_TIMEOUT = 120


def _run(args: list[str], timeout: int = _TIMEOUT) -> str:
    if not FFPROBE:
        return ""
    try:
        out = subprocess.run(
            [FFPROBE, "-v", "error", *args, "-of", "default=noprint_wrappers=1:nokey=1"],
            capture_output=True, text=True, timeout=timeout,
        )
        return out.stdout.strip()
    except Exception:
        return ""


def frame_pts(path: Path) -> list[float]:
    """读取视频全部帧的 pts_time（秒）。v3.0 按时间窗切分单集视频帧用。
    返回空列表表示无法读取（无 ffprobe / 文件异常）。"""
    if not FFPROBE or not Path(path).is_file():
        return []
    out = _run(["-select_streams", "v:0", "-show_frames", "-show_entries", "frame=pts_time", str(path)],
               timeout=600)
    if not out:
        return []
    pts: list[float] = []
    for line in out.splitlines():
        line = line.strip()
        if line:
            try:
                pts.append(float(line))
            except ValueError:  # noqa: PERF203
                continue
    return pts


def count_frames(path: Path) -> int | None:
    """mp4 视频帧数。返回 None 表示无法读取（无 ffprobe / 文件异常）。"""
    if not FFPROBE or not Path(path).is_file():
        return None
    # 1) 容器头 nb_frames（快）
    v = _run(["-select_streams", "v:0", "-show_entries", "stream=nb_frames", str(path)])
    if v.isdigit():
        return int(v)
    # 2) 慢速解码计数
    v = _run(["-count_frames", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames", str(path)], timeout=600)
    if v.isdigit():
        return int(v)
    return None


def probe_resolution(path: Path) -> tuple[int, int] | None:
    """视频分辨率 (w, h)，读不到返回 None。"""
    if not FFPROBE or not Path(path).is_file():
        return None
    v = _run(["-select_streams", "v:0", "-show_entries", "stream=width,height", str(path)])
    parts = v.split()
    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        return int(parts[0]), int(parts[1])
    return None