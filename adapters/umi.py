"""UMI / 手持采集 适配器：手持夹爪 + 头端视频 → 标准 v2.1（夹爪位姿动作通道）。

源布局（v1 简化约定，真实 UMI 原始包到位后在此校准解析器）：
    <src>/config.json                 {task, fps, cam, robot}
    <src>/demo_0001/cam_c0.mp4        第一视角视频（GoPro/采集相机）
    <src>/demo_0001/gripper_pose.csv  t,pos_x,pos_y,pos_z,quat_w,quat_x,quat_y,quat_z,open
    <src>/demo_0002/...               每 demo 一集

动作通道：夹爪位姿 8 维（3 平移 + 4 单位四元数 + 1 开度），即「夹爪位姿 = EEF」约定。
action = 同帧记录位姿（真实样本对齐后可用未来帧/差分策略调整）。
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

CSV_COLS = ["t", "pos_x", "pos_y", "pos_z", "quat_w", "quat_x", "quat_y", "quat_z", "open"]


def detect(src: Path) -> tuple[bool, str]:
    src = Path(src)
    if not (src / "config.json").is_file():
        return False, "缺 config.json（源目录标记文件）"
    demos = sorted(src.glob("demo_*"))
    if not demos:
        return False, "未发现 demo_* 目录"
    return True, f"{len(demos)} 个 demo"


def ingest(src: Path, out_root: Path, cfg: dict) -> tuple[Path, dict]:
    """解析 UMI 原始产物 → 标准 v2.1 数据集。返回 (输出目录, report)。"""
    src = Path(src)
    meta_cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    task = cfg.get("task") or meta_cfg.get("task", "umi_task")
    fps = float(cfg.get("fps") or meta_cfg.get("fps", 30))
    robot = cfg.get("robot") or meta_cfg.get("robot", "umi_gripper")
    cam = meta_cfg.get("cam", "cam_c0")

    demos = sorted(src.glob("demo_*"))
    episodes: list[dict] = []
    cam_spec: dict | None = None
    warnings: list[str] = []

    for i, demo in enumerate(demos):
        vid = demo / f"{cam}.mp4"
        pose_csv = demo / "gripper_pose.csv"
        if not vid.is_file() or not pose_csv.is_file():
            warnings.append(f"{demo.name}: 缺 {cam}.mp4 或 gripper_pose.csv，跳过")
            continue
        spec = common.probe_video(vid)
        if spec is None:
            warnings.append(f"{demo.name}: 视频不可读，跳过")
            continue
        cam_spec = {"width": spec["width"], "height": spec["height"], "fps": spec["fps"] or fps}

        df_raw = pd.read_csv(pose_csv)
        # 确保列齐
        for c in CSV_COLS[1:]:
            if c not in df_raw.columns:
                raise ValueError(f"{pose_csv.name} 缺列 {c}（要求 {CSV_COLS}）")
        n = len(df_raw)
        if spec["frames"] and spec["frames"] != n:
            warnings.append(f"{demo.name}: 视频帧数 {spec['frames']} ≠ csv 行数 {n}（以视频为准裁剪行数）")
        n = spec["frames"] if spec["frames"] else n
        df_raw = df_raw.head(n)

        pos = df_raw[["pos_x", "pos_y", "pos_z"]].to_numpy(np.float64)
        quat = df_raw[["quat_w", "quat_x", "quat_y", "quat_z"]].to_numpy(np.float64)
        opn = df_raw["open"].to_numpy(np.float64)
        state = common.gripper_pose_arrays(pos, quat, opn)
        df = common.build_frame_df(i, n, fps, state, state)  # action = 记录位姿
        episodes.append({"ep": i, "df": df, "videos": {cam: vid}})

    if not episodes:
        raise RuntimeError("UMI 源目录没有可用的 demo（见 warnings）")

    out = out_root / f"{task}_{robot}_{_stamp(meta_cfg)}_ingest"
    source_meta = {
        "source_type": "umi", "adapter": "umi.v1",
        "action_space": {"kind": "gripper_pose", "frame": "gripper",
                         "dims": "pos3+quat4+open1", "note": "夹爪位姿=EEF 约定，待真实样本校准"},
    }
    common.write_v21(out, episodes, robot_type=robot, fps=fps, task=task,
                     cam_specs={cam: cam_spec or {}}, source_meta=source_meta)

    report = {
        "source": "umi", "episodes": len(episodes), "frames": sum(len(e["df"]) for e in episodes),
        "cams": list({cam}), "action_channel": "gripper_pose(8D)", "output": str(out),
        "warnings": warnings,
    }
    return out, report


def _stamp(meta_cfg: dict) -> str:
    import datetime
    return datetime.datetime.now().strftime("%m%d")
