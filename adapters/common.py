"""INGEST 公共库：标准 v2.1 写出、视频探测、报告。
所有 adapter 复用此处的写出函数，保证产物与既有 01/02/03/05/07/08/登记完全兼容。
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipe.lib import dataset_io  # noqa: E402

# 夹爪/EEF 位姿动作列（带 gripper 前缀 → 03 关节检查自动豁免）
GRIPPER_COLS = [
    "pos_x", "pos_y", "pos_z",
    "quat_w", "quat_x", "quat_y", "quat_z",
    "open",
]

KEY_COLS = ["episode_index", "index", "timestamp", "frame_index", "task_index"]


def probe_video(path: Path) -> dict | None:
    """读视频宽/高/帧数（cv2 兜底；ffprobe 优先不依赖，避免无 ffprobe 时挂）。"""
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or None
    cap.release()
    return {"width": w, "height": h, "frames": n, "fps": fps}


def make_timestamps(n: int, fps: float) -> np.ndarray:
    """近期真实时刻的等间隔时间戳（登记卡日期才可读）。"""
    dt = 1.0 / fps
    t0 = float(time.time()) - (n - 1) * dt
    return t0 + np.arange(n) * dt


def build_frame_df(ep: int, n: int, fps: float, state: dict[str, np.ndarray],
                   action: dict[str, np.ndarray]) -> pd.DataFrame:
    """组装单个 episode 的 parquet 行数据。

    state/action: 列名（不含前缀）→ 数值数组。前缀由 write_v21 统一加，
    或这里直接给全名（含 observation.state./action. 前缀）。
    """
    cols = {
        "episode_index": np.full(n, ep, dtype=np.int64),
        "index": np.arange(n, dtype=np.int64),
        "timestamp": make_timestamps(n, fps),
        "frame_index": np.arange(n, dtype=np.int64),
        "task_index": np.zeros(n, dtype=np.int64),
    }
    for k, v in state.items():
        full = k if k.startswith("observation.") else f"observation.state.{k}"
        cols[full] = np.asarray(v, dtype=np.float64)
    for k, v in action.items():
        full = k if k.startswith("action.") else f"action.{k}"
        cols[full] = np.asarray(v, dtype=np.float64)
    return pd.DataFrame(cols)


def gripper_pose_arrays(pos: np.ndarray, quat: np.ndarray, opening: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """把 pos(3)/quat(4)/open 打包成 gripper_* 命名数组（与 03 豁免约定一致）。"""
    cols = {}
    for i, ax in enumerate("xyz"):
        cols[f"gripper_pos_{ax}"] = pos[:, i]
    for i, q in enumerate("wxyz"):
        cols[f"gripper_quat_{q}"] = quat[:, i]
    opening = np.ones(len(pos)) if opening is None else opening
    cols["gripper_open"] = opening
    return cols


def write_v21(out: Path, episodes: list[dict], *, robot_type: str, fps: float,
              task: str, cam_specs: dict[str, dict], source_meta: dict | None = None) -> Path:
    """把 episode 列表写成标准 v2.1 数据集。返回 out。

    episodes 每项: {ep, df, videos: {cam_key: Path|None}}
    cam_specs: {cam_key: {width, height, fps}}
    """
    out = Path(out)
    (out / "meta").mkdir(parents=True, exist_ok=True)
    (out / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    total_frames = 0
    ep_meta: list[dict] = []
    ep_stats: list[dict] = []
    used_cams: dict[str, dict] = {}

    for item in episodes:
        ep = int(item["ep"])
        df = item["df"]
        n = len(df)
        total_frames += n
        df.to_parquet(out / "data" / "chunk-000" / f"episode_{ep:06d}.parquet", index=False)

        # 视频：按需复制到 videos/chunk-000/<cam>/episode_%06d.mp4
        vids = item.get("videos") or {}
        for cam, src in vids.items():
            if not src or not Path(src).is_file():
                continue
            vdir = out / "videos" / "chunk-000" / cam
            vdir.mkdir(parents=True, exist_ok=True)
            dst = vdir / f"episode_{ep:06d}.mp4"
            if dst != src:
                shutil.copy2(src, dst)
            used_cams[cam] = cam_specs.get(cam) or {"width": 0, "height": 0, "fps": fps}

        ep_meta.append({
            "episode_index": ep,
            "tasks": [0],
            "length": n,
            "data_path": f"data/chunk-000/episode_{ep:06d}.parquet",
            "videos_path": "videos/chunk-000",
        })
        ep_stats.append(dataset_io.compute_episode_stats(df, ep))

    features = {c: {"dtype": "video"} for c in used_cams}
    info = {
        "codebase_version": "v2.1",
        "robot_type": robot_type,
        "fps": float(fps),
        "total_episodes": len(episodes),
        "total_frames": int(total_frames),
        "features": features,
        "videos": {c: {"width": int(s.get("width") or 0), "height": int(s.get("height") or 0),
                       "fps": float(s.get("fps") or fps)} for c, s in used_cams.items()},
    }
    if source_meta:
        info["source_meta"] = source_meta

    (out / "meta" / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "meta" / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": task}, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "meta" / "episodes.jsonl").write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in ep_meta) + "\n", encoding="utf-8")
    (out / "meta" / "episodes_stats.jsonl").write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in ep_stats) + "\n", encoding="utf-8")
    return out
