#!/usr/bin/env python3
"""生成 UMI 源合成样例（非标准布局，用于 INGEST 演示/自测，无真实数据时）。
布局与 adapters/umi.py 约定一致：demo_*/cam_c0.mp4 + gripper_pose.csv。
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def frame(ep: int, idx: int, w: int, h: int) -> np.ndarray:
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :, 0] = (ep * 40 + idx * 3) % 256
    img[:, :, 1] = 160
    img[:, :, 2] = (idx * 9) % 256
    block = 8
    for y in range(0, h, block):
        for x in range(0, w, block):
            if ((x // block) + (y // block)) % 2:
                img[y:y + block, x:x + block] = 30
    # 移动的"目标"方块（模拟抓取物靠近夹爪）
    cx = int(w * 0.5 + 0.3 * w * np.sin(idx / 20.0))
    cy = int(h * 0.4 + 0.25 * h * np.cos(idx / 25.0))
    cv2.rectangle(img, (cx - 8, cy - 8), (cx + 8, cy + 8), (220, 220, 40), -1)
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--task", default="pick_umi_cup")
    ap.add_argument("--res", default="128x96")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    w, h = (int(x) for x in args.res.lower().split("x"))

    for ep in range(args.episodes):
        d = out / f"demo_{ep:04d}"
        d.mkdir(parents=True, exist_ok=True)
        vw = cv2.VideoWriter(str(d / "cam_c0.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (w, h))
        n = args.frames + ep * 5
        tt = np.arange(n) / args.fps
        ph = np.linspace(0, 2 * np.pi, n)
        pos = np.stack([0.1 + 0.02 * np.sin(ph), 0.15 + 0.02 * np.cos(ph),
                        0.02 + 0.05 * ph / ph[-1]], axis=1)
        quat = np.zeros((n, 4))
        quat[:, 3] = np.cos(ph * 0.2 / 2)
        quat[:, 2] = np.sin(ph * 0.2 / 2)
        opn = np.clip(1 - ph / ph[-1], 0, 1)
        for i in range(n):
            vw.write(frame(ep, i, w, h))
        vw.release()
        df = pd.DataFrame({"t": tt, **{f"pos_{a}": pos[:, j] for j, a in enumerate("xyz")},
                           **{f"quat_{q}": quat[:, j] for j, q in enumerate("wxyz")},
                           "open": opn})
        df.to_csv(d / "gripper_pose.csv", index=False)

    (out / "config.json").write_text(json.dumps(
        {"source_type": "umi", "task": args.task, "fps": args.fps, "cam": "cam_c0",
         "robot": "umi_gripper", "note": "合成样例：布局模拟 UMI 手持采集产物"}, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] UMI 源合成样例: {out}（{args.episodes} demo，每 demo 视频+夹爪轨迹 csv）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
