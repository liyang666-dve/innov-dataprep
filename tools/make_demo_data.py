#!/usr/bin/env python3
"""生成"假" LeRobot v2.1 数据集，用于克隆后自测（无需真实数据）。

默认含刻意的脏数据，用于验证 01/02 的检出能力：
  - ep 0: 1 个重复时间戳 + 1 处 3× 间隔（丢帧窗口）
  - ep 1: 状态列 2 个 NaN
  - ep 2: 视频比 parquet 多 1 帧（对齐 mismatch）
加 --clean 可生成干净数据集。

用法:
    python3 tools/make_demo_data.py --out demo_data/arx_demo_0901_1500
    python3 tools/make_demo_data.py --out demo_data/clean_demo --clean
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

JOINT_NAMES = [f"left_{i}" for i in range(7)] + [f"right_{i}" for i in range(7)]


def make_info(robot_type: str, fps: int, total_episodes: int, total_frames: int, cams: dict, res: tuple[int, int]) -> dict:
    return {
        "codebase_version": "2.1",
        "robot_type": robot_type,
        "fps": fps,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "features": {},
        "videos": {c: {"width": res[0], "height": res[1], "fps": fps} for c in cams},
    }


def make_frame(ep: int, idx: int, w: int, h: int) -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 0] = (ep * 30 + idx * 3) % 256
    img[:, :, 1] = (idx * 7) % 256
    img[:, :, 2] = 200
    # 简单棋盘格，方便肉眼/算法确认帧确实在动
    block = 8
    for y in range(0, h, block):
        for x in range(0, w, block):
            if ((x // block) + (y // block)) % 2:
                img[y:y + block, x:x + block] = 40
                img[y:y + block, x:x + block, 0] += idx % 5
    return img


def write_videos(ds: Path, cams: list[str], ep: int, n_rows: int, fps: int, res: tuple[int, int],
                 extra_frames: int = 0) -> bool:
    try:
        import cv2
    except ImportError:
        print("[warn] 未安装 opencv，跳过视频生成（视频帧核对将报缺失）")
        return False
    w, h = res
    for cam in cams:
        vdir = ds / "videos" / "chunk-000" / cam
        vdir.mkdir(parents=True, exist_ok=True)
        vpath = vdir / f"episode_{ep:06d}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(str(vpath), fourcc, float(fps), (w, h))
        for i in range(n_rows + extra_frames):
            vw.write(make_frame(ep, i, w, h))
        vw.release()
    return True


def write_episode(ds: Path, ep: int, n: int, fps: int, task_index: int,
                  artifacts: str, res: tuple[int, int]) -> int:
    """写出单个 episode 的 parquet + 视频。返回该集行数。"""
    dt = 1.0 / fps
    t0 = float(time.time()) - (n - 1) * dt  # 近期真实时刻，登记卡日期才可读
    if artifacts == "dup_gap":
        dts = [dt] * (n - 1)
        dts[10] = 0.0          # 重复时间戳
        dts[20] = 3.0 * dt     # 丢帧窗口
        ts = t0 + np.cumsum(np.concatenate([[0.0], dts]))
    else:
        ts = t0 + np.arange(n) * dt

    t = np.linspace(0, 2 * np.pi, n)
    state = np.stack([0.5 + 0.3 * np.sin(t + k) for k in range(len(JOINT_NAMES))], axis=1)
    action = np.roll(state, -3, axis=0)  # 未来 3 帧的关节角
    action[-3:, :] = action[-3, :]

    if artifacts == "nan":
        state[5, 2] = np.nan
        state[12, 2] = np.nan

    cols = {}
    for i, jn in enumerate(JOINT_NAMES):
        cols[f"observation.state.{jn}"] = state[:, i]
        cols[f"action.{jn}"] = action[:, i]

    df_rows = {
        "episode_index": np.full(n, ep, dtype=np.int64),
        "index": np.arange(n, dtype=np.int64),
        "timestamp": ts,
        "frame_index": np.arange(n, dtype=np.int64),
        "task_index": np.full(n, task_index, dtype=np.int64),
        **cols,
    }
    import pandas as pd
    df = pd.DataFrame(df_rows)
    ddir = ds / "data" / "chunk-000"
    ddir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ddir / f"episode_{ep:06d}.parquet", index=False)

    extra = 1 if artifacts == "video_extra" else 0
    write_videos(ds, cams := ["front", "left_wrist"], ep, n, fps, res, extra_frames=extra)
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description="生成假 LeRobot v2.1 数据集（自测用）")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--frames", type=int, default=36)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--task", default="pick_and_place")
    ap.add_argument("--robot", default="bi_arx_x5")
    ap.add_argument("--res", default="64x48", help="视频分辨率 WxH")
    ap.add_argument("--clean", action="store_true", help="生成干净数据（无脏数据注入）")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        print(f"[!] {out} 已存在，先删除再生成")
        import shutil
        shutil.rmtree(out)
    (out / "meta").mkdir(parents=True, exist_ok=True)

    w, h = (int(x) for x in args.res.lower().split("x"))
    cams = ["front", "left_wrist"]
    task_index = 0

    artifacts_seq = ["dup_gap", "nan", "video_extra"] if not args.clean else ["", "", ""]
    total_frames = 0
    ep_meta = []
    for ep in range(args.episodes):
        n = args.frames + (ep if args.clean else 0)  # clean 模式各集帧数一致
        n = write_episode(out, ep, n, args.fps, task_index,
                          artifacts_seq[ep % len(artifacts_seq)], (w, h))
        total_frames += n
        ep_meta.append({
            "episode_index": ep, "tasks": [task_index], "length": n,
            "data_path": f"data/chunk-000/episode_{ep:06d}.parquet",
            "videos_path": "videos/chunk-000",
        })

    (out / "meta" / "info.json").write_text(
        json.dumps(make_info(args.robot, args.fps, args.episodes, total_frames, cams, (w, h)),
                   ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "meta" / "tasks.jsonl").write_text(
        json.dumps({"task_index": task_index, "task": args.task}, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "meta" / "episodes.jsonl").write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in ep_meta) + "\n", encoding="utf-8")

    mode = "干净" if args.clean else "含脏数据(重复戳/NaN/视频错帧)"
    print(f"[OK] 假数据集已生成: {out}  ({mode})")
    print(f"      {args.episodes} 集 / {total_frames} 帧 / {args.fps}fps / 任务 {args.task} / 机型 {args.robot}")
    print("      下一步: python3 pipe/01_inspect.py --input %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())