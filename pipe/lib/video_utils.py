"""视频工具：优先系统 ffprobe，缺省时用 opencv 兜底读帧数 / 分辨率。

策略：先读容器头里的 nb_frames（秒回）；拿不到再 -count_frames 全量解码。
ffprobe 与 cv2 都不可用时返回 None/空（调用方自行降级 WARN）。
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

FFPROBE: str | None = shutil.which("ffprobe") or shutil.which("ffprobe.exe")
_TIMEOUT = 120


def _cv2():
    """懒加载 opencv（可选依赖），缺省返回 None。"""
    try:
        import cv2  # noqa: PLC0415
        return cv2
    except Exception:
        return None


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
    if not Path(path).is_file():
        return None
    if FFPROBE:
        # 1) 容器头 nb_frames（快）
        v = _run(["-select_streams", "v:0", "-show_entries", "stream=nb_frames", str(path)])
        if v.isdigit():
            return int(v)
        # 2) 慢速解码计数
        v = _run(["-count_frames", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames", str(path)],
                 timeout=600)
        if v.isdigit():
            return int(v)
    # 3) opencv 兜底（无 ffprobe 的机器也能核对视频帧数）
    cv2 = _cv2()
    if cv2 is not None:
        try:
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                return None
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if n > 0:
                return n
            # 容器头没有帧数：逐帧解码计数
            cap = cv2.VideoCapture(str(path))
            cnt = 0
            while True:
                ok, _ = cap.read()
                if not ok:
                    break
                cnt += 1
            cap.release()
            return cnt
        except Exception:
            return None
    return None


def probe_resolution(path: Path) -> tuple[int, int] | None:
    """视频分辨率 (w, h)，读不到返回 None。"""
    if not Path(path).is_file():
        return None
    if FFPROBE:
        v = _run(["-select_streams", "v:0", "-show_entries", "stream=width,height", str(path)])
        parts = v.split()
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            return int(parts[0]), int(parts[1])
    cv2 = _cv2()
    if cv2 is not None:
        try:
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                return None
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if w and h:
                return w, h
        except Exception:
            return None
    return None