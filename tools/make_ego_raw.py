#!/usr/bin/env python3
"""生成 Ego 源合成样例（第一视角视频 + IMU，无机器人动作）。
布局与 adapters/ego.py 约定一致：ep*/head.mp4 + imu.csv。
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--task", default="ego_pour_water")
    ap.add_argument("--res", default="160x120")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    w, h = (int(x) for x in args.res.lower().split("x"))

    for ep in range(args.episodes):
        d = out / f"ep{ep:03d}"
        d.mkdir(parents=True)
        n = args.frames + ep * 8
        vw = cv2.VideoWriter(str(d / "head.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (w, h))
        for i in range(n):
            img = np.full((h, w, 3), (int(90 + 20 * np.sin(i / 30)), 120, 140), np.uint8)
            # 模拟第一视角：画面中央"手/桌"动态
            cv2.circle(img, (w // 2 + int(10 * np.sin(i / 12)), h // 2 + int(8 * np.cos(i / 14))),
                       min(w, h) // 6, (200, 190, 160), -1)
            vw.write(img)
        vw.release()
        tt = np.arange(n) / args.fps
        acc = np.stack([0.05 * np.sin(tt / 5), 9.8 + 0.1 * np.sin(tt / 8),
                        -0.03 * np.cos(tt / 6)], axis=1)
        gyro = np.stack([0.1 * np.cos(tt / 4), 0.05 * np.sin(tt / 3), 0.2 * np.sin(tt / 5)], axis=1)
        pd.DataFrame({"t": tt, **{f"acc_{a}": acc[:, j] for j, a in enumerate("xyz")},
                      **{f"gyro_{a}": gyro[:, j] for j, a in enumerate("xyz")}}).to_csv(
            d / "imu.csv", index=False)

    (out / "config.json").write_text(json.dumps(
        {"source_type": "ego", "task": args.task, "fps": args.fps, "cam": "head",
         "robot": "ego_human", "note": "合成样例：模拟头戴第一视角 + IMU"}, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Ego 源合成样例: {out}（{args.episodes} episode，视频+imu；无动作通道=重定向待接入）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
