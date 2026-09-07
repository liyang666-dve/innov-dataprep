"""仿真导出适配器：仿真引擎导出的轨迹 + 渲染视频 → 标准 v2.1（关节动作通道）。

源布局（v1 简化约定，对齐 Isaac/MuJoCo 常见导出形态：轨迹 csv + 每集渲染视频）：
    <src>/config.json              {task, fps, cam, robot, joints}
    <src>/episode_0000/render.mp4  仿真渲染视频（任意命名相机键见 config.cam）
    <src>/episode_0000/traj.csv    t,left_0..left_6,right_0..right_6（关节角，14 列）
    ...

动作通道：关节空间（与 ARX/innov 双臂口径一致）——仿真有 ground truth 关节角，
转换零损耗；这是与 UMI/ego 最大的不同：不需要动作恢复/重定向。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from adapters import common  # noqa: E402


def detect(src: Path) -> tuple[bool, str]:
    src = Path(src)
    if not (src / "config.json").is_file():
        return False, "缺 config.json"
    eps = sorted(src.glob("episode_*"))
    if not eps:
        return False, "未发现 episode_* 目录"
    return True, f"{len(eps)} 个 episode"


def ingest(src: Path, out_root: Path, cfg: dict) -> tuple[Path, dict]:
    src = Path(src)
    meta_cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    task = cfg.get("task") or meta_cfg.get("task", "sim_task")
    fps = float(cfg.get("fps") or meta_cfg.get("fps", 30))
    robot = cfg.get("robot") or meta_cfg.get("robot", "innov_arm")
    cam = meta_cfg.get("cam", "render")
    joints = meta_cfg.get("joints") or [f"left_{i}" for i in range(7)] + [f"right_{i}" for i in range(7)]

    eps = sorted(src.glob("episode_*"))
    episodes: list[dict] = []
    cam_spec: dict | None = None
    warnings: list[str] = []

    for i, ep_dir in enumerate(eps):
        vid = ep_dir / f"{cam}.mp4"
        traj = ep_dir / "traj.csv"
        if not vid.is_file() or not traj.is_file():
            warnings.append(f"{ep_dir.name}: 缺 {cam}.mp4 或 traj.csv，跳过")
            continue
        spec = common.probe_video(vid)
        if spec is None:
            warnings.append(f"{ep_dir.name}: 视频不可读，跳过")
            continue
        cam_spec = {"width": spec["width"], "height": spec["height"], "fps": spec["fps"] or fps}

        df_t = pd.read_csv(traj)
        n = len(df_t)
        if spec["frames"] and spec["frames"] != n:
            warnings.append(f"{ep_dir.name}: 视频 {spec['frames']} 帧 ≠ 轨迹 {n} 行（截齐）")
            n = min(spec["frames"], n)
        state = {}
        for jn in joints:
            if jn not in df_t.columns:
                raise ValueError(f"{traj.name} 缺关节列 {jn}")
            state[jn] = df_t[jn].head(n).to_numpy(np.float64)
        df = common.build_frame_df(i, n, fps, state, state)
        episodes.append({"ep": i, "df": df, "videos": {cam: vid}})

    if not episodes:
        raise RuntimeError("Sim 源目录没有可用 episode")

    out = common.write_v21(
        out_root / f"{task}_{robot}_{_stamp()}_ingest", episodes,
        robot_type=robot, fps=fps, task=task,
        cam_specs={cam: cam_spec or {}},
        source_meta={
            "source_type": "sim", "adapter": "sim.v1",
            "action_space": {"kind": "joint", "joints": joints,
                             "note": "仿真 ground-truth 关节角，转换零损耗"},
        },
    )
    report = {
        "source": "sim", "episodes": len(episodes),
        "frames": sum(len(e["df"]) for e in episodes),
        "cams": [cam], "action_channel": f"joint({len(joints)}D)", "output": str(out),
        "warnings": warnings,
    }
    return out, report


def _stamp() -> str:
    import datetime
    return datetime.datetime.now().strftime("%m%d")
