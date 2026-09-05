"""Ego / 穿戴采集 适配器：第一视角视频 + IMU → 标准 v2.1（视频 + 元数据，动作待重定向）。

源布局（v1 简化约定）：
    <src>/config.json          {task, fps, cam, robot}
    <src>/ep001/head.mp4       头戴第一视角视频
    <src>/ep001/imu.csv        t,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z（旁路保留）

动作通道说明：ego 采集的是"人的操作"，不含机器人可执行动作。本 v1 只做容器标准化
（视频 + 时间戳 + meta），动作列留空并在 source_meta 标记 retarget_pending——
真实数据到位后，若采集端能同步夹爪/手部位姿（如穿戴夹爪手柄），可切到 gripper_pose 通道。
"""
from __future__ import annotations

import json
import shutil
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
    eps = sorted(src.glob("ep*"))
    if not eps:
        return False, "未发现 ep* 目录"
    return True, f"{len(eps)} 个 episode"


def ingest(src: Path, out_root: Path, cfg: dict) -> tuple[Path, dict]:
    src = Path(src)
    meta_cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    task = cfg.get("task") or meta_cfg.get("task", "ego_task")
    fps = float(cfg.get("fps") or meta_cfg.get("fps", 30))
    robot = cfg.get("robot") or meta_cfg.get("robot", "ego_human")
    cam = meta_cfg.get("cam", "head")

    eps = sorted(src.glob("ep*"))
    episodes: list[dict] = []
    cam_spec: dict | None = None
    warnings: list[str] = []

    for i, ep_dir in enumerate(eps):
        vid = ep_dir / f"{cam}.mp4"
        if not vid.is_file():
            warnings.append(f"{ep_dir.name}: 缺 {cam}.mp4，跳过")
            continue
        spec = common.probe_video(vid)
        if spec is None:
            warnings.append(f"{ep_dir.name}: 视频不可读，跳过")
            continue
        cam_spec = {"width": spec["width"], "height": spec["height"], "fps": spec["fps"] or fps}
        n = spec["frames"] or 0
        # 仅视频 + 时间轴：无 state/action 数值列（03 关节 QC 自动跳过）
        df = common.build_frame_df(i, n, fps, {}, {})
        episodes.append({"ep": i, "df": df, "videos": {cam: vid}})

        imu = ep_dir / "imu.csv"
        if imu.is_file():
            episodes[-1]["imu_src"] = imu

    if not episodes:
        raise RuntimeError("Ego 源目录没有可用 episode")

    out = common.write_v21(
        out_root / f"{task}_{robot}_{_stamp()}_ingest", episodes,
        robot_type=robot, fps=fps, task=task,
        cam_specs={cam: cam_spec or {}},
        source_meta={
            "source_type": "ego", "adapter": "ego.v1",
            "action_space": {"kind": "retarget_pending",
                             "note": "第一视角无直接机器人动作；待夹爪/手部动作重定向通道"},
        },
    )
    # IMU 旁路保留：meta/sensors/episode_%06d/imu.csv
    for item in episodes:
        imu_src = item.get("imu_src")
        if imu_src and imu_src.is_file():
            d = out / "meta" / "sensors" / f"episode_{item['ep']:06d}"
            d.mkdir(parents=True, exist_ok=True)
            shutil.copy2(imu_src, d / "imu.csv")

    report = {
        "source": "ego", "episodes": len(episodes),
        "frames": sum(len(e["df"]) for e in episodes),
        "cams": [cam], "action_channel": "none (retarget_pending)", "output": str(out),
        "warnings": warnings,
    }
    return out, report


def _stamp() -> str:
    import datetime
    return datetime.datetime.now().strftime("%m%d")
