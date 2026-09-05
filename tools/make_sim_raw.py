#!/usr/bin/env python3
"""生成 Sim 源合成样例（仿真导出形态：渲染视频 + 关节轨迹 csv）。
布局与 adapters/sim.py 约定一致：episode_*/render.mp4 + traj.csv。
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

JOINTS = [f"left_{i}" for i in range(7)] + [f"right_{i}" for i in range(7)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--task", default="sim_stack_blocks")
    ap.add_argument("--res", default="160x120")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    w, h = (int(x) for x in args.res.lower().split("x"))

    for ep in range(args.episodes):
        d = out / f"episode_{ep:04d}"
        d.mkdir(parents=True)
        n = args.frames + ep * 10
        vw = cv2.VideoWriter(str(d / "render.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (w, h))
        for i in range(n):
            img = np.zeros((h, w, 3), np.uint8)
            img[:, :] = (int(30 + 10 * np.sin(i / 20)), 40, 60)
            # 方块 + 臂的简单示意
            cv2.rectangle(img, (int(w * (0.2 + 0.5 * np.abs(np.sin(i / 40)))), h // 3),
                          (int(w * (0.3 + 0.5 * np.abs(np.sin(i / 40)))) + 10, h // 2), (120, 200, 120), -1)
            vw.write(img)
        vw.release()
        ph = np.linspace(0, 4 * np.pi, n)
        data = {"t": np.arange(n) / args.fps}
        for j, jn in enumerate(JOINTS):
            data[jn] = 0.4 * np.sin(ph + j * 0.7)
        pd.DataFrame(data).to_csv(d / "traj.csv", index=False)

    (out / "config.json").write_text(json.dumps(
        {"source_type": "sim", "task": args.task, "fps": args.fps, "cam": "render",
         "robot": "innov_arm", "joints": JOINTS,
         "note": "合成样例：模拟仿真引擎导出（render 视频 + ground-truth 关节轨迹）"},
        ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Sim 源合成样例: {out}（{args.episodes} episode，视频+14 关节轨迹）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
