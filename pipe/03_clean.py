#!/usr/bin/env python3
"""03 清洗/质检：对 LeRobot v2.1 数据集逐集检查并软标记（只读，绝不改数据）。

铁律：本脚本不删除/修改任何数据文件。坏集只在报告中标记 exclude，
真正的过滤发生在 05_merge（按 episode_disposition.csv 排除）。

产出（写入 <输入父目录>/<名字>_clean/）:
  - qc_report.md            人类可读报告
  - episode_disposition.csv 每集一行: verdict=keep|exclude + 理由（05 合并消费）
  - summary.json            机器可读

用法:
    python3 pipe/03_clean.py --input <dataset> [--input ...] [--config config.yaml]
          [--blur] [--no-video-check]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipe.lib import dataset_io, report, video_utils  # noqa: E402

GRIPPER_HINTS = ("gripper", "open", "close", "finger")


# ---------------------------------------------------------------- 配置
DEFAULT_QC = {
    "min_duration_s": 3.0,        # 短于 -> exclude
    "max_duration_s": 120.0,      # 长于 -> warn（不 exclude）
    "fps_deviation": 0.15,        # 实际帧率与标称偏差 -> exclude
    "max_drop_ratio": 0.05,       # 丢帧估计/行数 -> exclude
    "joint_limits_rad": 3.3,      # |关节角| 超限 -> exclude
    "joint_jump_rad": 0.8,        # 相邻帧跳变 -> exclude
    "stuck_s": 0.4,               # 关节零方差持续 -> exclude
    "blur_laplacian_thr": 3.0,    # 帧 Laplacian 方差阈值（绝对）
    "blur_bad_ratio": 0.10,       # 低于阈值的帧占比 -> exclude
    "blur_sample_frames": 200,    # 每相机抽帧上限（控成本）
}


def load_qc(args: argparse.Namespace) -> dict:
    qc = dict(DEFAULT_QC)
    cfg_path = Path(args.config)
    if cfg_path.is_file():
        import yaml
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        qc.update(cfg.get("qc") or {})
    if args.blur:
        qc["check_blur"] = True
    return qc


# ---------------------------------------------------------------- 单集检查
def _gripper_like(col: str) -> bool:
    return any(h in col.lower() for h in GRIPPER_HINTS)


def _state_arrays(df: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """返回 (非夹爪关节列, 矩阵)；缺列时为空。"""
    cols = [c for c in df.columns if c.startswith("observation.state.") and not _gripper_like(c)]
    cols = [c for c in cols if df[c].dtype.kind in "fc"]
    if not cols:
        return [], np.zeros((len(df), 0))
    return cols, df[cols].to_numpy(dtype=np.float64)


def check_episode(df: pd.DataFrame, meta_info: dict, qc: dict, nominal_fps: float,
                  video_summary: dict[str, dict], check_joints: bool = True) -> dict[str, Any]:
    """对单个 episode 执行全部检查。返回 verdict/reasons/warnings 等。

    check_joints=False 时跳过关节限位/跳变/卡死检查（v3.0 用：保持与数组列形态的
    v2.1 行为一致——v2.1 单数组列不展开，本检查本就不触发，故 v3.0 也不误判持物静止）。
    """
    n = len(df)
    exclude: list[str] = []
    warn: list[str] = []

    # --- 状态/动作维度一致性
    state_feats = {c[len("observation.state."):] for c in df.columns if c.startswith("observation.state.")}
    act_feats = {c[len("action."):] for c in df.columns if c.startswith("action.")}
    if state_feats and act_feats and state_feats != act_feats:
        exclude.append(f"状态-动作特征不一致: state={sorted(state_feats)[:5]}… action={sorted(act_feats)[:5]}…")

    # --- 时间
    has_ts = "timestamp" in df.columns
    if has_ts:
        t = df["timestamp"].to_numpy(dtype=np.float64)
        t = t[np.isfinite(t)]
        dur = float(t[-1] - t[0]) if t.size >= 2 else 0.0
    else:
        dur = n / nominal_fps
    if dur < qc["min_duration_s"]:
        exclude.append(f"时长过短 {dur:.1f}s < {qc['min_duration_s']}s")
    elif dur > qc["max_duration_s"]:
        warn.append(f"时长偏长 {dur/60:.1f}min > {qc['max_duration_s']}s")
    if has_ts and t.size >= 2:
        d = np.diff(t)
        actual_fps = (n - 1) / dur if dur > 0 else np.nan
        dev = abs(actual_fps - nominal_fps) / nominal_fps if nominal_fps else np.nan
        if dev > qc["fps_deviation"]:
            exclude.append(f"实际帧率 {actual_fps:.1f} 偏离标称 {nominal_fps} {dev*100:.0f}%")
        med = float(np.median(d))
        if med > 0:
            dropped = int(np.sum(np.maximum(0.0, np.round(d / med) - 1)))
            if n and dropped / n > qc["max_drop_ratio"]:
                exclude.append(f"丢帧 {dropped}/{n} ({dropped/n*100:.0f}%) > {qc['max_drop_ratio']*100:.0f}%")
            n_back = int(np.sum(d < 0))
            n_dup = int(np.sum(d <= 0))
            if n_back:
                exclude.append(f"时间戳回退 {n_back} 处")
            elif n_dup:
                warn.append(f"重复时间戳 {n_dup} 处")

    # --- NaN / Inf（状态+动作全量数值列）
    num_cols = [c for c in df.columns if c not in dataset_io.KEY_COLUMNS and df[c].dtype.kind in "fc"]
    bad = 0
    if num_cols:
        arr = df[num_cols].to_numpy(dtype=np.float64)
        bad = int(np.isnan(arr).sum() + np.isinf(arr).sum())
    if bad:
        exclude.append(f"NaN/Inf {bad} 个")

    # --- 关节：限位 / 跳变 / 卡死（v3.0 关闭，见 check_joints）
    joint_cols, J = (_state_arrays(df) if check_joints else ([], np.zeros((len(df), 0))))
    if joint_cols and J.shape[1]:
        with np.errstate(all="ignore"):
            lim = qc["joint_limits_rad"]
            if lim and (np.nanmax(np.abs(J)) if J.size else 0) > lim:
                worst = int(np.nanargmax(np.abs(J))) if J.size else 0
                exclude.append(f"关节超限位 {np.nanmax(np.abs(J)):.2f}rad > {lim} (列 {joint_cols[worst]})")
            jd = np.abs(np.diff(J, axis=0))
            thr = qc["joint_jump_rad"]
            if jd.size:
                mj = np.nanmax(jd) if jd.size else 0.0
                if mj > thr:
                    col = int(np.nanargmax(jd) % jd.shape[1]) if jd.size else 0
                    exclude.append(f"关节跳变 {mj:.2f}rad > {thr} (列 {joint_cols[col]})")
            # 卡死：任一行在 diff==0 连续最长
            eq = np.diff(J, axis=0) == 0
            med_dt = np.nanmedian(d) if has_ts and t.size >= 2 else 1.0 / nominal_fps
            stuck_len_s = 0.0
            if eq.size:
                # 每列连续相同行数最大值 -> 秒
                runs = []
                for c in range(eq.shape[1]):
                    cnt = 0
                    mx = 0
                    for v in eq[:, c]:
                        cnt = cnt + 1 if v else 0
                        mx = max(mx, cnt)
                    runs.append(mx)
                stuck_len_s = max(runs, default=0) * med_dt
                if stuck_len_s > qc["stuck_s"]:
                    col = int(np.argmax(runs)) if runs else 0
                    exclude.append(f"关节卡死 {stuck_len_s:.2f}s > {qc['stuck_s']}s (列 {joint_cols[col]})")

    # --- 视频
    for cam, v in video_summary.items():
        if v.get("missing"):
            exclude.append(f"视频缺失 {cam}")
        elif v.get("frames") is not None:
            diff = v["frames"] - n
            if abs(diff) == 1:
                warn.append(f"视频帧数差1可复核 {cam}: {v['frames']} vs {n}")
            elif abs(diff) > 1:
                exclude.append(f"视频帧数不符 {cam}: {v['frames']} vs {n} (差 {diff:+d})")

    return {
        "n_rows": n, "duration_s": round(dur, 3),
        "verdict": "exclude" if exclude else "keep",
        "reasons_exclude": exclude,
        "reasons_warn": warn,
    }


# ---------------------------------------------------------------- 视频模糊（可选，重）
def blur_check(video_summary: dict, qc: dict) -> dict[str, Any]:
    """每相机抽帧算 Laplacian 方差。cv2/文件缺失时返回空。"""
    try:
        import cv2
    except ImportError:
        return {"skipped": "no cv2"}
    out: dict[str, Any] = {}
    n = 0
    total_bad = 0.0
    for cam, v in video_summary.items():
        if v.get("missing") or not v.get("path"):
            continue
        cap = cv2.VideoCapture(str(v["path"]))
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not total or total < 2:
            cap.release()
            continue
        idxs = np.linspace(0, total - 1, min(qc["blur_sample_frames"], int(total)), dtype=int)
        variances = []
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            variances.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
        cap.release()
        if variances:
            ratio = sum(1 for x in variances if x < qc["blur_laplacian_thr"]) / len(variances)
            n += len(variances)
            total_bad = max(total_bad, ratio)
            out[cam] = {"sampled": len(variances), "bad_ratio": round(ratio, 3),
                        "median_var": round(float(np.median(variances)), 2)}
    if not out:
        return {"skipped": "no readable videos"}
    out["_worst_bad_ratio"] = round(total_bad, 3)
    out["_bad"] = total_bad > qc["blur_bad_ratio"]
    return out


# ---------------------------------------------------------------- 主流程
def run_dataset(ds: Path, qc: dict, args: argparse.Namespace) -> dict:
    if dataset_io.detect_dataset(ds)[0] == "v3.0":
        return _run_dataset_v30(ds, qc, args)
    meta = dataset_io.read_meta(ds)
    info = meta["info"]
    nominal = float(info.get("fps") or args.fps or 30.0)
    cams = dataset_io.camera_layout(ds, info)
    eps_paths = dataset_io.discover_episodes(ds)
    do_videos = not args.no_video_check
    do_blur = qc.get("check_blur") and do_videos
    if do_videos and video_utils.FFPROBE is None:
        print(f"[WARN] ffprobe 不可用，{ds.name} 的视频帧数核对跳过（sudo apt install ffmpeg）")
    if do_blur:
        try:
            import cv2  # noqa: F401
        except ImportError:
            print(f"[WARN] 未安装 opencv，{ds.name} 的模糊检查跳过（pip install opencv-python-headless）")

    ep_rows = []
    for p in eps_paths:
        ep_idx = dataset_io.episode_index(p)
        df = pd.read_parquet(p)
        n_rows = len(df)
        # 视频摘要（帧数/缺失/mismatch）与模糊共用一次探测
        rel = str(p.relative_to(ds))
        src = Path(rel)
        chunk_dir = src.parent.name if src.parent.name != "data" else "chunk-000"
        vsum: dict[str, dict] = {}
        for cam in cams:
            cand = None
            for c2 in (ds / "videos" / chunk_dir / cam / (src.name.replace(".parquet", ".mp4")),
                       ds / "videos" / "chunk-000" / cam / (src.name.replace(".parquet", ".mp4"))):
                if c2.is_file():
                    cand = c2
                    break
            if cand is None:
                vsum[cam] = {"frames": None, "missing": True, "path": None}
            else:
                fr = video_utils.count_frames(cand)
                vsum[cam] = {"frames": fr, "missing": False, "path": cand}
        q = check_episode(df, info, qc, nominal, vsum)
        if do_blur:
            b = blur_check(vsum, qc)
            if b.get("_bad"):
                q["reasons_exclude"].append(f"模糊帧占比 {b['_worst_bad_ratio']*100:.0f}% > "
                                            f"{qc['blur_bad_ratio']*100:.0f}%")
                q["verdict"] = "exclude"
        ep_rows.append({"episode": ep_idx, "n_rows": n_rows, "_qc": q, "videos": vsum})

    ep_rows.sort(key=lambda x: x["episode"])
    n_keep = sum(1 for e in ep_rows if e["_qc"]["verdict"] == "keep")
    n_excl = len(ep_rows) - n_keep

    out_root = Path(args.out) if args.out else dataset_io.new_stage_dir(ds, "clean")
    return _finalize_qc(ds, out_root, ep_rows, nominal, n_keep, n_excl)


def _finalize_qc(ds: Path, out_root: Path, ep_rows: list[dict], nominal: float,
                 n_keep: int | None = None, n_excl: int | None = None) -> dict:
    """写 episode_disposition.csv / summary.json / qc_report.md（v2.1/v3.0 共用）。"""
    if n_keep is None:
        n_keep = sum(1 for e in ep_rows if e["_qc"]["verdict"] == "keep")
    if n_excl is None:
        n_excl = len(ep_rows) - n_keep
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for e in ep_rows:
        r = {"episode": e["episode"], "n_rows": e["n_rows"], "verdict": e["_qc"]["verdict"],
             "duration_s": e["_qc"]["duration_s"],
             "reasons_exclude": " | ".join(e["_qc"]["reasons_exclude"]) or "-",
             "reasons_warn": " | ".join(e["_qc"]["reasons_warn"]) or "-"}
        for cam, v in e["videos"].items():
            r[f"video_{cam}"] = ("缺失" if v["missing"] else
                                 (f"{v['frames']}" if v["frames"] is not None else "n/a"))
        rows.append(r)
    report.write_csv(out_root / "episode_disposition.csv", rows)

    summary = {
        "dataset": ds.name, "path": str(ds), "nominal_fps": nominal,
        "n_episodes": len(ep_rows), "n_keep": n_keep, "n_exclude": n_excl,
        "excluded_episodes": [e["episode"] for e in ep_rows if e["_qc"]["verdict"] == "exclude"],
        "disposition_csv": str(out_root / "episode_disposition.csv"),
    }
    report.write_json(out_root / "summary.json", summary)

    md = [
        f"# 质检报告: {ds.name}", "",
        f"- 通过 keep: {n_keep} / {n_excl} 排除",
        f"- 排除集: {summary['excluded_episodes'] or '无'}",
        "", "## 逐集", "",
        "| ep | 行数 | 时长s | 结论 | 原因 |", "|---|---|---|---|---|",
    ]
    for e in ep_rows:
        q = e["_qc"]
        why = "; ".join(q["reasons_exclude"] + [f"⚠ {w}" for w in q["reasons_warn"]]) or "✓"
        md.append(f"| {e['episode']} | {e['n_rows']} | {q['duration_s']:.2f} | {q['verdict']} | {why} |")
    md += ["", "## 说明", "", "- 软标记：本脚本未修改任何数据文件；05 合并时按 episode_disposition.csv 排除 exclude 集。", ""]
    report.write_md(out_root / "qc_report.md", md)
    return summary


def _run_dataset_v30(ds: Path, qc: dict, args: argparse.Namespace) -> dict:
    """v3.0 清洗/质检：逐集 df 已展开成 v2.1 风格列，check_episode 复用；
    逐集视频按 meta/episodes 的 from/to_timestamp 在每机位大 mp4 上切窗核对。
    模糊检查(blur)对 v3.0 跳过（无单集视频文件，整形 mp4 过重）。"""
    meta = dataset_io.read_meta(ds)
    info = meta["info"]
    nominal = float(info.get("fps") or args.fps or 30.0)
    cams = dataset_io._v3_cameras(ds, info)
    eps_meta = dataset_io._v3_episodes_meta(ds)
    do_videos = not args.no_video_check
    qc["check_blur"] = False  # v3.0 跳过模糊
    if args.blur:
        print(f"[WARN] {ds.name} 为 v3.0，模糊检查自动跳过（无单集视频文件）")

    if do_videos and video_utils.FFPROBE is None:
        print(f"[WARN] ffprobe 不可用，{ds.name} 的视频帧数核对跳过（sudo apt install ffmpeg）")
    print(f"[INFO] {ds.name} 为 v3.0，关节限位/跳变/卡死检查跳过（与数组列 v2.1 行为一致）")
    vid_by_ep = (dataset_io._v3_episode_video_summary(ds, eps_meta, cams)
                 if do_videos else {})
    ep_rows = []
    for ep, df in dataset_io.iter_v3_episodes(ds):
        vsum = vid_by_ep.get(ep, {})
        q = check_episode(df, info, qc, nominal, vsum, check_joints=False)
        ep_rows.append({"episode": ep, "n_rows": len(df), "_qc": q, "videos": vsum})

    ep_rows.sort(key=lambda x: x["episode"])
    n_keep = sum(1 for e in ep_rows if e["_qc"]["verdict"] == "keep")
    n_excl = len(ep_rows) - n_keep
    out_root = Path(args.out) if args.out else dataset_io.new_stage_dir(ds, "clean")
    return _finalize_qc(ds, out_root, ep_rows, nominal, n_keep, n_excl)


def main() -> int:
    ap = argparse.ArgumentParser(description="LeRobot v2.1 清洗/质检（软标记，只读）")
    ap.add_argument("--input", action="append", required=True)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--no-video-check", action="store_true")
    ap.add_argument("--blur", action="store_true", help="开启模糊帧检查（需 cv2，逐视频抽帧，较慢）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    qc = load_qc(args)
    rc = 0
    for s in args.input:
        ds = Path(s)
        kind, reason = dataset_io.detect_dataset(ds)
        if kind not in ("v2.1", "v3.0"):
            print(f"[跳过] {ds.name} 不是 v2.1/v3.0 数据集: {reason}")
            rc = 1
            continue
        summary = run_dataset(ds, qc, args)
        print(f"[OK] {summary['dataset']}: keep {summary['n_keep']} / exclude {summary['n_exclude']} -> "
              f"{summary['disposition_csv']}")
        if summary["excluded_episodes"]:
            print(f"      排除集: {summary['excluded_episodes']}")
    return rc


if __name__ == "__main__":
    sys.exit(main())