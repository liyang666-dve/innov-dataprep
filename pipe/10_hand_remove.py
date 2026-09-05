#!/usr/bin/env python3
"""10 去手处理：把视频里出现的人手/手臂抹掉，防止训练模型学到"人手"（也兼顾隐私）。

为什么需要：
  - UMI 手持 / ego 头戴采集的第一视角数据，人手几乎必然入镜；
  - VLA/扩散策略若从画面学到"人手=动作"，部署到真机（没有手）时会失效或学偏；
  - 部分场景还涉及操作者隐私（公开数据集前要去手）。

两档模式：
  轻档（本阶段默认，CPU 可跑）：MediaPipe Hands 检测手 → 手掌包围盒 + 从手腕沿
      手臂方向延伸的"粗线"覆盖可见前臂 → mask 羽化 → 盖板（模糊 blur / 涂暗 dark）。
      定位是"够用 + 快 + 零 GPU"，启发式延伸对手持/ego 这类"手臂从画面边缘伸入"
      的画面覆盖良好；手臂横穿画面中段、姿态复杂的，会盖不干净——那种请用重档。
  重档（预留接口，需公司机 + GPU）：--inpainter e2fgvi|propainter。本模块负责
      产出 mask 帧序列供修复模型消费（export_masks 已实现）；调用外部修复模型的
      命令模板见文件底部 INPAINTER_CALL 约定，当前选择非 off 会提示并按轻档回退。

用法:
    python3 pipe/10_hand_remove.py --input <数据集目录> [--input <更多数据集>]
          [--mode blur|dark] [--cams cam_c0] [--out <输出父目录>] [--force]
    python3 pipe/10_hand_remove.py --video <单个.mp4> [--out <输出目录>]
    # 开发自测：不依赖真实人手，强制按"底部人手区"注入遮罩跑通盖板+重编码链路
    python3 pipe/10_hand_remove.py --input <数据集> --force-test-hand

铁律：只写 --out 副本，绝不修改源数据集；检测不到手的视频原样拷贝（不重编码、无画质损失）。

依赖：mediapipe==0.10.14（内置手部模型，无需下载；pip install mediapipe==0.10.14）
       + opencv-python（env 已有）。
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# MediaPipe 内部 TF Lite 日志很吵且无信息量，压到 WARNING 以下
os.environ.setdefault("GLOG_minloglevel", "2")
logging.getLogger("mediapipe").setLevel(logging.ERROR)

# ---------------------------------------------------------------- 常量
DEFAULT_MIN_CONF = 0.5     # MediaPipe 手部检测置信度
DEFAULT_SAMPLE = 5         # 预检抽帧间隔（每 N 帧检 1 帧；无手则整视频原样拷贝）
DEFAULT_ARM_EXT = 1.6      # 手臂延伸 = 手腕→中指根 距离 × 倍数（覆盖可见前臂长度）
MASK_FEATHER_SIGMA = 4.0   # 遮罩羽化（高斯）sigma，避免盖板硬边
HAND_MIN_RATIO = 0.06      # 手部包围盒小于该图像占比视为误检（像素太少不可信）

# 重档修复模型调用约定（占位）：mask 序列导出后按此模板调外部仓库。
# {video}  原始视频路径   {maskdir}  export_masks 导出的逐帧 mask 目录  {out}  输出视频
INPAINTER_CALL = {
    "e2fgvi": "python -m e2fgvi.infer --video {video} --mask_dir {maskdir} --out {out}",
    "propainter": "python inference_propainter.py --video {video} --mask {maskdir} --output {out}",
}


# ---------------------------------------------------------------- 检测器
class HandDetector:
    """MediaPipe Hands（static_image_mode 逐帧）。solutions 内置权重，无需下载 .task。"""

    def __init__(self, min_conf: float = DEFAULT_MIN_CONF) -> None:
        try:
            from mediapipe.python.solutions import hands as mp_hands  # noqa: PLC0415
        except Exception as e:  # noqa: BLE001
            raise SystemExit(
                "[ERROR] 需要 mediapipe==0.10.14：pip install \"mediapipe==0.10.14\"\n"
                f"        （当前导入失败: {type(e).__name__}: {e}）"
            ) from e
        self._hands = mp_hands.Hands(
            static_image_mode=True, max_num_hands=2,
            min_detection_confidence=min_conf,
        )

    def detect(self, frame_bgr: np.ndarray) -> list[dict]:
        """返回手列表：{rect:(x0,y0,x1,y1) 原图坐标, lm:21点, score}。"""
        out: list[dict] = []
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_bgr.shape[:2]
        try:
            res = self._hands.process(rgb)
        except Exception:  # noqa: BLE001  单帧异常不致命
            return out
        if not res.multi_hand_landmarks:
            return out
        for lm_list in res.multi_hand_landmarks:
            lm = [(p.x * w, p.y * h) for p in lm_list.landmark]
            xs = [p[0] for p in lm]
            ys = [p[1] for p in lm]
            bw, bh = max(xs) - min(xs), max(ys) - min(ys)
            # 手部占比过小视为误检（小图噪声）
            if max(bw, bh) < HAND_MIN_RATIO * max(w, h):
                continue
            out.append({
                "rect": (min(xs), min(ys), max(xs), max(ys)),
                "lm": lm, "score": 1.0,
            })
        return out


# ---------------------------------------------------------------- 遮罩与盖板
def _hand_mask(frame: np.ndarray, lm: list[tuple[float, float]],
               arm_ext: float) -> np.ndarray:
    """手部 bbox + 手腕→手臂方向粗线，构成 0/255 mask（覆盖手 + 进入画面的前臂）。"""
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    xs = [p[0] for p in lm]
    ys = [p[1] for p in lm]
    x0, y0, x1, y1 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    # 手 bbox 外扩 12% 覆盖指缝/手掌边缘
    mx, my = int(0.12 * bw), int(0.12 * bh)
    cv2.rectangle(mask, (max(0, x0 - mx), max(0, y0 - my)),
                  (min(w, x1 + mx), min(h, y1 + my)), 255, -1)
    # 手臂延伸：wrist(0) -> 中指根(9) 反方向为手臂走向
    wrist = np.array(lm[0], dtype=float)
    mid_mcp = np.array(lm[9], dtype=float)
    d = wrist - mid_mcp
    nd = np.linalg.norm(d)
    if nd > 1e-6:
        u = d / nd
        arm_len = nd * arm_ext
        # 沿方向推进到图像边界内
        t = 1.0
        end = wrist + u * arm_len
        while (end[0] < 0 or end[0] >= w or end[1] < 0 or end[1] >= h) and t > 0.02:
            t *= 0.85
            end = wrist + u * (arm_len * t)
        if t > 0.02:
            thick = max(2, int(bw * 0.8))
            cv2.line(mask, (int(wrist[0]), int(wrist[1])),
                     (int(end[0]), int(end[1])), 255, thick)
    return mask


def cover(frame: np.ndarray, mask: np.ndarray, mode: str) -> np.ndarray:
    """按 mask 羽化盖板：blur=高斯模糊填充 / dark=压暗（0.12 亮度）。"""
    if mode == "dark":
        fill = (frame * 0.12).astype(np.uint8)
    else:  # blur
        sigma = max(9.0, min(frame.shape[:2]) / 22.0)
        fill = cv2.GaussianBlur(frame, (0, 0), sigmaX=sigma, sigmaY=sigma)
    mf = mask.astype(np.float32) / 255.0
    if MASK_FEATHER_SIGMA > 0:
        mf = cv2.GaussianBlur(mf, (0, 0), sigmaX=MASK_FEATHER_SIGMA)
    mf = mf[..., None]
    return (frame.astype(np.float32) * (1 - mf)
            + fill.astype(np.float32) * mf).astype(np.uint8)


def _force_hand_mask(frame: np.ndarray) -> np.ndarray:
    """--force-test-hand：把画面底部 38% 当作"人手区"（开发自测用，无真实手也能
    跑通 mask→盖板→重编码整链路；MediaPipe 真实识别留给真数据验证）。"""
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[int(h * 0.62):, :] = 255
    return mask


# ---------------------------------------------------------------- 视频处理
def process_video(src: Path, dst: Path, mode: str, detector: HandDetector,
                  force_test: bool = False, sample: int = DEFAULT_SAMPLE,
                  arm_ext: float = DEFAULT_ARM_EXT) -> dict:
    """处理单个视频。返回 {hand_frames, total, action: rewritten|copy}。

    copy = 预检无手，未重编码（dst 不存在，由上层决定原样拷贝）；调用方负责无手时拷贝源文件。
    """
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        return {"action": "error", "reason": f"打不开视频 {src}", "total": 0, "hand_frames": 0}
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # 预检：抽帧看有没有手（force_test 视为有手）
    has_hand = force_test
    if not force_test:
        idx = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if idx % sample == 0 and detector.detect(fr):
                has_hand = True
                break
            idx += 1
        cap.release()
        if not has_hand:
            return {"action": "copy", "total": total, "hand_frames": 0,
                    "reason": "预检未检测到手，原样拷贝（不重编码）"}
        cap = cv2.VideoCapture(str(src))

    # 全帧处理并重编码到临时文件
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    tmp = dst.with_suffix(dst.suffix + ".tmp.mp4")
    writer = cv2.VideoWriter(str(tmp), fourcc, fps, (w, h))
    hand_frames = 0
    n = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if force_test:
            m = _force_hand_mask(fr)
            fr = cover(fr, m, mode)
            hand_frames += 1
        else:
            hands = detector.detect(fr)
            if hands:
                m = np.zeros((h, w), dtype=np.uint8)
                for hd in hands:
                    m = cv2.bitwise_or(m, _hand_mask(fr, hd["lm"], arm_ext))
                fr = cover(fr, m, mode)
                hand_frames += 1
        writer.write(fr)
        n += 1
    cap.release()
    writer.release()
    if n == 0:
        tmp.unlink(missing_ok=True)
        return {"action": "error", "reason": "视频无可读帧", "total": 0, "hand_frames": 0}
    tmp.replace(dst)
    return {"action": "rewritten", "total": total, "hand_frames": hand_frames}


def find_videos(ds: Path, cams: list[str] | None) -> list[Path]:
    """收集数据集内全部 mp4（videos/**/*.mp4，兼容官方与自产布局），可按相机名过滤。"""
    vids = sorted(ds.rglob("*.mp4"))
    if cams:
        vids = [v for v in vids
                if any(c.lower() in v.as_posix().lower() for c in cams)]
    return vids


# ---------------------------------------------------------------- 数据集级
def run_dataset(ds: Path, out_root: Path, mode: str, detector: HandDetector,
                cams: list[str] | None, arm_ext: float, sample: int,
                force: bool, force_test: bool, inpainter: str) -> dict:
    """数据集副本式处理：meta/data 原样拷贝，视频按需重写。返回报告 dict。"""
    out_ds = out_root / f"{ds.name}_nohand"
    if out_ds.exists():
        if not force:
            raise SystemExit(f"[ERROR] 输出已存在: {out_ds}\n        换目录或用 --force 覆盖")
        shutil.rmtree(out_ds)
    shutil.copytree(ds, out_ds)   # 先整树拷贝（parquet/meta/未处理视频都不动）

    videos = find_videos(ds, cams)
    rep: dict[str, list] = {"videos": [], "rewritten": 0, "copied": 0, "failed": 0}
    t0 = time.time()
    for v in videos:
        rel = v.relative_to(ds).as_posix()
        dst = out_ds / v.relative_to(ds)
        entry = {"video": rel}
        r = process_video(v, dst, mode, detector, force_test=force_test,
                          sample=sample, arm_ext=arm_ext)
        entry.update({k: r[k] for k in ("action", "total", "hand_frames") if k in r})
        if r.get("reason"):
            entry["reason"] = r["reason"]
        if r["action"] == "rewritten":
            rep["rewritten"] += 1
        elif r["action"] == "copy":
            rep["copied"] += 1
        else:
            rep["failed"] += 1
        rep["videos"].append(entry)
        print(f"  {'✓' if r['action'] != 'error' else '✗'} {rel:<48} "
              f"{entry.get('action','?'):<10} 手帧 {entry.get('hand_frames', '-')}/{entry.get('total','-')}"
              + (f"  ({entry['reason']})" if entry.get("reason") else ""))
    rep["seconds"] = round(time.time() - t0, 1)
    report = {
        "tool": "10_hand_remove", "time": datetime.now().isoformat(timespec="seconds"),
        "source": str(ds), "output": str(out_ds),
        "mode": mode, "arm_ext": arm_ext, "inpainter_requested": inpainter,
        "inpainter_note": "非 off 时本版本回退轻档；重档需公司机按 INPAINTER_CALL 约定接入",
        "seconds": rep["seconds"],
        "totals": {"videos": len(videos), "rewritten": rep["rewritten"],
                   "copied": rep["copied"], "failed": rep["failed"]},
        "videos": rep["videos"],
    }
    (out_ds / "meta").mkdir(exist_ok=True)
    (out_ds / "meta" / "hand_remove_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report


# ---------------------------------------------------------------- 入口
def main() -> int:
    ap = argparse.ArgumentParser(prog="10_hand_remove.py", description="去手处理（轻档盖板 / 重档预留）")
    ap.add_argument("--input", action="append", default=None, help="数据集目录（可多次）")
    ap.add_argument("--video", default=None, help="单个视频文件（快速试用，--input 二选一）")
    ap.add_argument("--out", default=None, help="输出父目录（默认数据集同父目录 / 视频同目录）")
    ap.add_argument("--mode", choices=["blur", "dark"], default="blur")
    ap.add_argument("--cams", default=None, help="只处理含这些相机名的视频，逗号分隔（默认全部）")
    ap.add_argument("--min-conf", type=float, default=DEFAULT_MIN_CONF)
    ap.add_argument("--sample", type=int, default=DEFAULT_SAMPLE)
    ap.add_argument("--arm-ext", type=float, default=DEFAULT_ARM_EXT)
    ap.add_argument("--inpainter", choices=["off", "e2fgvi", "propainter"], default="off",
                    help="重档修复（预留，需公司机 GPU + 外部仓库）")
    ap.add_argument("--force-test-hand", dest="force_test", action="store_true",
                    help="开发自测：把画面底部当人手区注入遮罩，无真实手也能验证链路")
    ap.add_argument("--force", action="store_true", help="覆盖已存在的输出目录")
    args = ap.parse_args()
    if bool(args.input) == bool(args.video):
        ap.error("--input 与 --video 二选一")

    detector = HandDetector(min_conf=args.min_conf)
    cams = [c.strip() for c in args.cams.split(",") if c.strip()] if args.cams else None
    print(f"[10_hand_remove] mode={args.mode} cams={cams or 'all'} "
          f"arm_ext={args.arm_ext} inpainter={args.inpainter}")

    if args.inpainter != "off":
        print(f"[i] --inpainter {args.inpainter} 为重档预留：需要公司机部署 E2FGVI/ProPainter 后\n"
              f"    按调用约定接入（{INPAINTER_CALL[args.inpainter]}）。本版本先按轻档盖板处理。")

    reports = []
    if args.video:
        v = Path(args.video).resolve()
        out_dir = Path(args.out).resolve() if args.out else v.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        dst = out_dir / f"{v.stem}_nohand.mp4"
        r = process_video(v, dst, args.mode, detector, force_test=args.force_test,
                          sample=args.sample, arm_ext=args.arm_ext)
        if r["action"] == "copy":
            shutil.copy2(v, dst)
        print(f"[10_hand_remove] {v.name} → {dst.name}  {r['action']} "
              f"手帧 {r.get('hand_frames', 0)}/{r.get('total', '?')}")
        (out_dir / f"{v.stem}_hand_remove.json").write_text(
            json.dumps({"tool": "10_hand_remove", "video": str(v), "output": str(dst),
                        "time": datetime.now().isoformat(timespec="seconds"), **r},
                       ensure_ascii=False, indent=1), encoding="utf-8")
        return 0 if r["action"] != "error" else 1

    for ds_str in args.input:
        ds = Path(ds_str).resolve()
        if not ds.is_dir():
            print(f"[ERROR] 数据集目录不存在: {ds}")
            return 1
        out_root = Path(args.out).resolve() if args.out else ds.parent
        print(f"[10_hand_remove] 处理数据集: {ds.name}")
        rep = run_dataset(ds, out_root, args.mode, detector, cams,
                          args.arm_ext, args.sample, args.force,
                          args.force_test, args.inpainter)
        reports.append(rep)
        t = rep["totals"]
        print(f"[完成] {t['videos']} 视频 | 重写 {t['rewritten']} | 原样拷贝 {t['copied']} "
              f"| 失败 {t['failed']} | 耗时 {rep['seconds']}s | 输出 {rep['output']}")
    return 0 if all(r["totals"]["failed"] == 0 for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
