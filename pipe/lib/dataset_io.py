"""LeRobot v2.1 数据集读写与摘要。

约定布局（v2.1）:
    <dataset>/
    ├── meta/info.json            # robot_type / fps / features / total_* / videos(分辨率)
    ├── meta/tasks.jsonl          # {"task_index":0,"task":"..."}
    ├── meta/episodes.jsonl       # 可选
    ├── data/chunk-000/episode_000000.parquet
    └── videos/chunk-000/<cam>/episode_000000.mp4
"""
from __future__ import annotations

import json
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import video_utils

META_DIR = "meta"
DATA_DIR = "data"
VIDEO_DIR = "videos"
KEY_COLUMNS = ("episode_index", "index", "frame_index", "task_index", "timestamp")


# ---------------------------------------------------------------- 基础 IO
def detect_dataset(path: Path) -> tuple[str, str]:
    """识别目录类型。返回 (kind, reason)：
    kind ∈ {v2.1, v3.0, not_dataset}。
    - v2.1: meta/info.json + 至少 1 个 data/chunk-*/episode_*.parquet
    - v3.0: 有 meta/info.json 但结构像 v3.0（file-*.parquet / meta/episodes/）
    - not_dataset: 其他（带 reason）
    """
    path = Path(path)
    if not (path / META_DIR / "info.json").is_file():
        return "not_dataset", f"缺 {META_DIR}/info.json"
    # v3.0 特征：data/chunk-*/file-*.parquet 或 meta/episodes/ 目录
    if any((path / DATA_DIR).glob("chunk-*/file-*.parquet")) or (path / META_DIR / "episodes").is_dir():
        return "v3.0", "疑似 v3.0 数据集（当前管道处理 v2.1，请先用 06 转换器的逆向或换环境）"
    if not discover_episodes(path):
        return "not_dataset", f"未发现 {DATA_DIR}/chunk-*/episode_*.parquet（空目录/数据未落盘？）"
    return "v2.1", ""


def is_dataset_dir(path: Path) -> bool:
    return detect_dataset(path)[0] == "v2.1"


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_meta(ds: Path) -> dict[str, Any]:
    info = read_json(ds / META_DIR / "info.json")
    tasks = read_jsonl(ds / META_DIR / "tasks.jsonl")
    eps_meta = read_jsonl(ds / META_DIR / "episodes.jsonl")
    task_map = {t.get("task_index"): t.get("task") for t in tasks}
    return {"info": info, "tasks": tasks, "episodes_meta": eps_meta, "task_map": task_map}


def discover_episodes(ds: Path) -> list[Path]:
    base = ds / DATA_DIR
    pats = sorted(base.glob("chunk-*/episode_*.parquet"))
    if not pats:
        pats = sorted(base.glob("episode_*.parquet"))
    return pats


def episode_index(path: Path) -> int:
    m = re.search(r"episode_(\d+)", path.name)
    return int(m.group(1)) if m else -1


def camera_layout(ds: Path, meta_info: dict[str, Any]) -> dict[str, dict[str, int]]:
    """相机键 -> {w,h}。先读 meta 的 videos 信息，videos/ 目录兜底。"""
    cams: dict[str, dict[str, int]] = {}
    vinfo = meta_info.get("videos")
    if isinstance(vinfo, dict):
        for k, v in vinfo.items():
            if isinstance(v, dict) and isinstance(k, str):
                cams[k] = {"w": int(v.get("width") or 0), "h": int(v.get("height") or 0)}
    vdir = ds / VIDEO_DIR
    if vdir.is_dir():
        for chunk in sorted(vdir.glob("chunk-*")):
            for cam_dir in chunk.iterdir():
                if cam_dir.is_dir():
                    cams.setdefault(cam_dir.name, {"w": 0, "h": 0})
    return cams


# ---------------------------------------------------------------- 分集统计
def _nan_summary(df: pd.DataFrame) -> tuple[int, list[str]]:
    cols = [c for c in df.columns if c not in KEY_COLUMNS and df[c].dtype.kind in "fc"]
    if not cols:
        return 0, []
    bad = df[cols].isna()
    n = int(bad.values.sum())
    cols_with = [c for c in cols if bool(bad[c].any())]
    return n, cols_with[:10]


def _ts_stats(df: pd.DataFrame) -> dict[str, Any]:
    """时间戳纯审计（不改数据）。"""
    if "timestamp" not in df.columns:
        return {"has_timestamp": False}
    t = df["timestamp"].to_numpy(dtype=np.float64)
    t = t[np.isfinite(t)]
    if t.size < 2:
        return {"has_timestamp": True, "n": int(t.size), "first": None, "last": None,
                "duration_s": 0.0, "median_dt": None, "max_dt": 0.0,
                "n_dup": 0, "n_backward": 0, "n_gaps2x": 0, "dropped_est": 0}
    d = np.diff(t)
    med = float(np.median(d)) if d.size else 0.0
    dur = float(t[-1] - t[0])
    dropped = int(np.sum(np.maximum(0.0, np.round(d / med) - 1))) if med > 0 else 0
    return {
        "has_timestamp": True, "n": int(t.size),
        "first": float(t[0]), "last": float(t[-1]),
        "duration_s": round(dur, 4),
        "median_dt": round(med, 6),
        "max_dt": round(float(d.max()), 4),
        "n_dup": int(np.sum(d <= 0)),
        "n_backward": int(np.sum(d < 0)),
        "n_gaps2x": int(np.sum(d > 2 * med)) if med > 0 else 0,
        "dropped_est": dropped,
    }


def _video_check(ds: Path, parquet_rel: str, n_rows: int,
                 cams: dict[str, dict[str, int]]) -> dict[str, dict[str, Any]]:
    """每相机视频帧数与 parquet 行数核对（只审计不修改）。

    mismatch = 视频帧数 - parquet 行数；帧数读不到时为 None。
    """
    src = Path(parquet_rel)
    chunk_dir = src.parent.name if src.parent.name != "data" else "chunk-000"
    ep_name = src.name.replace(".parquet", ".mp4")
    out: dict[str, dict[str, Any]] = {}
    for cam in cams:
        found = None
        for cand in (ds / VIDEO_DIR / chunk_dir / cam / ep_name,
                     ds / VIDEO_DIR / "chunk-000" / cam / ep_name):
            if cand.is_file():
                found = cand
                break
        if found is None:
            out[cam] = {"frames": None, "missing": True, "mismatch": None}
        else:
            frames = video_utils.count_frames(found)
            out[cam] = {"frames": frames, "missing": False,
                        "mismatch": None if frames is None else frames - n_rows}
    return out


def summarize_episode(ds: Path, path: Path, fps_nominal: float,
                      cams: dict[str, dict[str, int]], meta_info: dict[str, Any],
                      check_videos: bool = True) -> dict[str, Any]:
    df = pd.read_parquet(path)
    n = len(df)
    ts = _ts_stats(df)
    nan_n, nan_cols = _nan_summary(df)

    ep: dict[str, Any] = {
        "episode": episode_index(path),
        "n_rows": n,
        "n_nan": nan_n,
        "nan_cols": nan_cols,
    }
    if ts.get("has_timestamp"):
        dur = ts["duration_s"]
        ep.update({
            "first_ts": ts["first"], "last_ts": ts["last"],
            "duration_s": dur,
            "fps_actual": round((n - 1) / dur, 3) if dur > 0 else None,
            "median_dt": ts["median_dt"], "max_dt": ts["max_dt"],
            "n_dup_ts": ts["n_dup"], "n_backward_ts": ts["n_backward"],
            "n_gaps2x": ts["n_gaps2x"], "dropped_est": ts["dropped_est"],
        })
    else:
        ep.update({
            "first_ts": None, "last_ts": None,
            "duration_s": round(n / fps_nominal, 4), "fps_actual": None,
            "median_dt": None, "max_dt": 0.0,
            "n_dup_ts": 0, "n_backward_ts": 0, "n_gaps2x": 0, "dropped_est": 0,
        })

    if check_videos:
        ep["videos"] = _video_check(ds, str(path.relative_to(ds)), n, cams)
    else:
        ep["videos"] = {}
    return ep


def summarize_dataset(ds: Path, fps_nominal: float = 30.0, check_videos: bool = True) -> dict[str, Any]:
    """整集摘要：元信息 + 逐集统计 + 汇总 + 问题列表。"""
    ds = Path(ds)
    if not is_dataset_dir(ds):
        raise ValueError(f"不是 LeRobot v2.1 数据集（缺 meta/info.json）: {ds}")
    meta = read_meta(ds)
    info = meta["info"]
    nominal = float(info.get("fps") or fps_nominal)
    cams = camera_layout(ds, info)
    eps_paths = discover_episodes(ds)
    episodes = [summarize_episode(ds, p, nominal, cams, info, check_videos) for p in eps_paths]
    episodes.sort(key=lambda e: e["episode"])

    total_frames = int(sum(e["n_rows"] for e in episodes))
    total_dur = float(sum(e.get("duration_s") or 0.0 for e in episodes))

    first_ts = min((e["first_ts"] for e in episodes if e.get("first_ts") is not None), default=None)
    last_ts = max((e["last_ts"] for e in episodes if e.get("last_ts") is not None), default=None)

    issues: list[str] = []
    for e in episodes:
        if e["n_dup_ts"] > 0:
            issues.append(f"ep{e['episode']}: {e['n_dup_ts']} 个重复/回退时间戳")
        if e["dropped_est"] > 0:
            issues.append(f"ep{e['episode']}: 估计丢帧 {e['dropped_est']} 帧")
        if e["n_nan"] > 0:
            issues.append(f"ep{e['episode']}: 状态/动作含 {e['n_nan']} 个 NaN（列: {','.join(e['nan_cols'][:4])}）")
        if e["fps_actual"] is not None and nominal > 0:
            dev = abs(e["fps_actual"] - nominal) / nominal
            if dev > 0.15:
                issues.append(f"ep{e['episode']}: 实际帧率 {e['fps_actual']} 与标称 {nominal} 偏差 {dev*100:.0f}%")
        for cam, v in e.get("videos", {}).items():
            if v.get("missing"):
                issues.append(f"ep{e['episode']}: 视频缺失 {cam}")
            elif v.get("frames") is not None and v["frames"] != e["n_rows"]:
                issues.append(f"ep{e['episode']}: {cam} 视频帧数 {v['frames']} ≠ parquet 行数 {e['n_rows']}")

    return {
        "path": str(ds),
        "name": ds.name,
        "format": "v2.1",
        "robot_type": str(info.get("robot_type") or "unk"),
        "fps_nominal": nominal,
        "task_names": [t.get("task") for t in meta["tasks"]],
        "cameras": cams,
        "n_episodes": len(episodes),
        "total_frames": total_frames,
        "duration_h": round(total_dur / 3600.0, 2),
        "min_date": _ts_to_date(first_ts),
        "max_date": _ts_to_date(last_ts),
        "episodes": episodes,
        "issues": issues,
    }


def _ts_to_date(ts: float | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).date().isoformat()


def _valid_mmdd(s: str) -> bool:
    if len(s) != 4 or not s.isdigit():
        return False
    m, d = int(s[:2]), int(s[2:])
    return 1 <= m <= 12 and 1 <= d <= 31


def _mmdd_of_yyyymmdd(s: str) -> str | None:
    """'20260901' -> '0901'；月份/日期非法返回 None。"""
    if len(s) != 8 or not s.isdigit():
        return None
    mm, dd = s[4:6], s[6:8]
    return mm + dd if _valid_mmdd(mm + dd) else None


_DATE_RE = re.compile(
    r"(?P<y8a>\d{8})-(?P<y8b>\d{8})"
    r"|(?P<y8>\d{8})"
    r"|(?P<m4a>\d{4})-(?P<m4b>\d{4})"
    r"|(?P<m4>\d{4})"
)


def parse_date_range(name: str) -> tuple[str, str] | None:
    """从目录名解析日期范围，返回 (start_MMDD, end_MMDD) 或 None。

    支持：MMDD / MMDD_HHMM / MMDD-MMDD / YYYYMMDD / YYYYMMDD-YYYYMMDD。
    规则：MMDD 必须月份 01-12、日期 01-31 才认账；解析不出/歧义返回 None，
    由调用方决定报错（绝不生成 0101 这类假日期）。
    """
    for m in _DATE_RE.finditer(name):
        if m.group("y8a"):
            a, b = _mmdd_of_yyyymmdd(m.group("y8a")), _mmdd_of_yyyymmdd(m.group("y8b"))
            if a and b:
                return a, b
        if m.group("y8"):
            a = _mmdd_of_yyyymmdd(m.group("y8"))
            if a:
                return a, a
        if m.group("m4a"):
            a, b = m.group("m4a"), m.group("m4b")
            if _valid_mmdd(a) and _valid_mmdd(b):
                return a, b
        if m.group("m4"):
            a = m.group("m4")
            if _valid_mmdd(a):
                return a, a
    return None


def compute_episode_stats(df: pd.DataFrame, episode_index: int) -> dict[str, Any]:
    """按真实 robodeploy v2.1 格式生成一行 episodes_stats.jsonl。

    {"episode_index": n, "stats": {feature: {min, max, mean, std, count}}}
    只统计数值特征列（跳过 KEY_COLUMNS），官方 v2.1→v3.0 转换器硬性需要此文件。
    """
    cols = [c for c in df.columns if c not in KEY_COLUMNS and df[c].dtype.kind in "fc"]
    stats: dict[str, Any] = {}
    for c in cols:
        s = df[c]
        cnt = int(s.count())
        if cnt == 0:
            continue
        stats[c] = {
            "min": float(s.min()), "max": float(s.max()),
            "mean": float(s.mean()), "std": float(s.std()), "count": cnt,
        }
    return {"episode_index": int(episode_index), "stats": stats}


def summarize_light(path: Path) -> dict[str, Any]:
    """轻量摘要（只读 meta + 数分集文件，不读 parquet/视频）。run.py 列表用。"""
    path = Path(path)
    kind, reason = detect_dataset(path)
    if kind != "v2.1":
        return {"path": str(path), "name": path.name, "kind": kind, "reason": reason}
    meta = read_meta(path)
    info = meta["info"]
    n_eps = len(discover_episodes(path))
    return {
        "path": str(path),
        "name": path.name,
        "kind": "v2.1",
        "reason": "",
        "robot_type": str(info.get("robot_type") or "unk"),
        "fps": float(info.get("fps") or 0.0),
        "n_episodes": n_eps,
        "tasks": [t.get("task") for t in meta["tasks"]],
    }