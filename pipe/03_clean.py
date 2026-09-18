#!/usr/bin/env python3
"""03 清洗/质检：对 LeRobot v2.1 数据集逐集检查并软标记（只读，绝不改数据）。

铁律：本脚本不删除/修改任何数据文件。坏集只在报告中标记 exclude，
真正的过滤发生在 05_merge（按 episode_disposition.csv 排除）。

数组列说明（2026-09 审计后修复）：真实 robodeploy v2.1 把 action / observation.state
存成"每行一个数组"的 object 列，历史实现按 dtype/点分前缀选列 → NaN/Inf、关节类检查
在真实数据上完全不触发。现改为先 expand_array_features() 展开成 action.0…N 再检查，
v2.1 与 v3.0 行为一致。关节类判定因数组列没有关节名（无法豁免夹爪）默认关闭，
只输出「关节观测量」供定标，加 --joints 才参与 keep/exclude。

产出（写入 <输入父目录>/<名字>_products/clean/）:
  - qc_report.md            人类可读报告（含关节观测量）
  - episode_disposition.csv 每集一行: verdict=keep|exclude + 理由（05 合并消费）
  - summary.json            机器可读（含 joint_observation）

用法:
    python3 pipe/03_clean.py --input <dataset> [--input ...] [--config config.yaml]
          [--blur] [--no-video-check] [--joints]
"""
from __future__ import annotations

import argparse
import hashlib
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

    # --- 动作/状态统计信号（review 级：只提示、绝不排除）---
    # 这几项对标 lerobot-doctor / RDA 的核心检查（零方差维、动作尖峰、僵死维、有效运动比）
    "zero_var_eps": 1e-6,           # 维度标准差小于它 -> 常量维（review）
    "action_spike_sigma": 8.0,      # 逐维 |Δ| 超过 8×该维整体标准差 -> 尖峰（review，对齐 lerobot-doctor）
    "action_spike_min_abs": 0.5,    # 且绝对跳变 >= 该值（rad）才算尖峰，避免平滑数据的抖动误报
    "stuck_dim_ratio": 0.8,         # 单维不变帧占比超过 -> 记为「僵死维」（info 级，报告里汇总）
    "idle_move_eps": 1e-3,          # 单帧全维位移小于它算静止
    "idle_ratio_warn": 0.90,        # 静止帧占比超过 -> review（有效运动比 <10%）
    "duration_outlier_sigma": 3.0,  # 时长偏离均值超过 3σ -> review
    "stats_drift_tol": 0.05,        # meta/episodes_stats 与实测均值偏差超过该比例 -> review
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


# ---------------------------------------------------------------- 统计信号（review 级）
def _signal_checks(df: "pd.DataFrame", qc: dict) -> tuple[list[str], dict]:
    """零方差维 / 动作尖峰 / 僵死维 / 有效运动比。

    纯统计、与本体无关，v2.1 与 v3.0、数组列（已展开）与点分列通用。
    判据（对齐 lerobot-doctor 的口径，2026-09 在真实 0730 三批上定标）：
      · 常量维：该维标准差 < zero_var_eps            -> review（通道死了/没记）
      · 尖峰  ：|Δ| > max(σ倍数×该维标准差, 绝对下限)  -> review（真实跳变）
      · 静止帧占比 > idle_ratio_warn                 -> review（几乎没动）
      · 僵死维（不变帧占比高）                        -> **info 级**，数据集汇总里列出
        （夹爪本来就常驻不动，逐集标 review 会把 100% 集都标上，等于没标）

    只产生 review 级原因，从不参与 exclude —— 阈值未定标前不能杀数据。
    """
    review: list[str] = []
    info: dict[str, Any] = {}
    k_sig = float(qc.get("action_spike_sigma") or 0)
    floor = float(qc.get("action_spike_min_abs") or 0)
    zero_eps = float(qc.get("zero_var_eps") or 0)
    stuck_thr = float(qc.get("stuck_dim_ratio") or 1.1)
    idle_warn = float(qc.get("idle_ratio_warn") or 1.1)
    idle_eps = float(qc.get("idle_move_eps") or 0)

    for prefix, tag in (("action", "action"), ("observation.state", "state")):
        cols = [c for c in df.columns if c.startswith(prefix + ".")]
        if not cols:
            continue
        X = df[cols].to_numpy(dtype=np.float64)
        if X.size:
            X = X[np.isfinite(X).all(axis=1)]
        if X.shape[0] < 2:
            continue
        with np.errstate(all="ignore"):
            info[f"{tag}_n_dims"] = int(X.shape[1])
            std = np.nanstd(X, axis=0)
            const = [i for i, v in enumerate(std) if v < zero_eps]
            info[f"{tag}_const_dims"] = const
            if const:
                review.append(f"{prefix} 常量维 {len(const)} 个 dim{const[:8]}"
                              f"{'…' if len(const) > 8 else ''}（标准差<{zero_eps:g}）")
            D = np.diff(X, axis=0)
            if not D.size:
                continue
            thr = np.maximum(k_sig * std, floor) if k_sig > 0 else np.full_like(std, np.inf)
            over = np.abs(D) > thr
            n_sp = int(over.sum())
            info[f"{tag}_spikes"] = n_sp
            if n_sp:
                idx = np.unravel_index(int(np.argmax(np.abs(D) - thr)), D.shape)
                if np.abs(D[idx]) > thr[idx[1]]:
                    review.append(f"{prefix} 尖峰 {n_sp} 处 (>max({k_sig:g}σ,{floor:g}rad))，"
                                  f"最坏 {abs(float(D[idx])):.3f} rad @dim{idx[1]}")
            ratio = (np.abs(D) < 1e-12).mean(axis=0)
            stuck = [i for i, r in enumerate(ratio) if r > stuck_thr]
            info[f"{tag}_stuck_dims"] = stuck
            step = np.linalg.norm(D, axis=1)
            idle = float((step < idle_eps).mean()) if step.size else 0.0
            info[f"{tag}_idle_ratio"] = round(idle, 4)
            info[f"{tag}_max_jump"] = round(float(np.nanmax(np.abs(D))), 4)
            if idle > idle_warn:
                review.append(f"{prefix} 静止帧占比 {idle:.0%} > {idle_warn:.0%}"
                              f"（有效运动比 {1 - idle:.0%}）")
    return review, info


def _fingerprint(df: "pd.DataFrame") -> str | None:
    """近重复集指纹：state/action 逐维均值+标准差的 2 位小数摘要。"""
    vec: list[float] = []
    for prefix in ("observation.state", "action"):
        cols = [c for c in df.columns if c.startswith(prefix + ".")]
        if not cols:
            continue
        X = df[cols].to_numpy(dtype=np.float64)
        if not X.size:
            continue
        with np.errstate(all="ignore"):
            vec += [round(float(v), 2) for v in np.nanmean(X, axis=0)]
            vec += [round(float(v), 2) for v in np.nanstd(X, axis=0)]
    if not vec:
        return None
    return hashlib.sha1(json.dumps(vec).encode()).hexdigest()[:12]


def _parse_meta_stats(ds: Path) -> dict[int, dict]:
    """读 meta/episodes_stats.jsonl（用于 stats 漂移检查）；缺失返回 {}。"""
    p = ds / "meta" / "episodes_stats.jsonl"
    if not p.is_file():
        return {}
    out: dict[int, dict] = {}
    try:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                out[int(r["episode_index"])] = r.get("stats") or {}
    except Exception:  # noqa: BLE001
        return {}
    return out


def _stats_drift(df: "pd.DataFrame", stats: dict, tol: float) -> str | None:
    """meta 里的 stats 均值与实测均值偏差（超过 tol×该维标准差 -> 提示）。"""
    if not stats or not tol:
        return None
    worst = 0.0
    worst_key = ""
    for prefix in ("observation.state", "action"):
        cols = [c for c in df.columns if c.startswith(prefix + ".")]
        st = stats.get(prefix)
        if not cols or not isinstance(st, dict):
            continue
        X = df[cols].to_numpy(dtype=np.float64)
        if not X.size:
            continue
        with np.errstate(all="ignore"):
            cur = np.nanmean(X, axis=0)
            ref = np.asarray(st.get("mean") or [], dtype=np.float64)
            scale = np.asarray(st.get("std") or [], dtype=np.float64)
        if ref.shape != cur.shape:
            continue
        scale = np.where(np.abs(scale) < 1e-9, 1.0, np.abs(scale))
        rel = np.abs(cur - ref) / scale
        if rel.size and float(np.nanmax(rel)) > worst:
            worst = float(np.nanmax(rel))
            worst_key = f"{prefix} dim{int(np.nanargmax(rel))}"
    if worst > tol:
        return f"stats 与实测不符（{worst_key} 偏差 {worst:.0%} > {tol:.0%}，meta/episodes_stats.jsonl 可能是旧值）"
    return None


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
    review: list[str] = []

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

    # --- 统计信号（零方差维/尖峰/僵死维/有效运动比）：只提示，不排除
    sig_review, signals = _signal_checks(df, qc)
    review += sig_review

    dims = {p: len([c for c in df.columns if c.startswith(p + ".")])
            for p in ("action", "observation.state")}
    verdict = "exclude" if exclude else ("review" if review else "keep")
    return {
        "n_rows": n, "duration_s": round(dur, 3),
        "verdict": verdict,
        "reasons_exclude": exclude,
        "reasons_review": review,
        "reasons_warn": warn,
        "signals": signals,
        "dims": dims,
        "fingerprint": _fingerprint(df),
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


# ---------------------------------------------------------------- 关节观测量（只测不判）
def _joint_observation(df: "pd.DataFrame", nominal_fps: float) -> dict:
    """数组列数据集上的关节观测量：只看数值，不参与 keep/exclude 判定。

    真实 v2.1/v3.0 的 action / observation.state 是"每行一个数组"，没有关节名，
    因此无法豁免夹爪维度 → 关节限位/跳变/卡死阈值必须先按本观测量定标，
    再用 --joints 打开判定（默认关闭，避免未经定标就误杀）。
    """
    cols, J = _state_arrays(df)
    if not cols or not J.size:
        return {}
    with np.errstate(all="ignore"):
        max_abs = float(np.nanmax(np.abs(J)))
        jd = np.abs(np.diff(J, axis=0))
        max_jump = float(np.nanmax(jd)) if jd.size else 0.0
        eq = np.diff(J, axis=0) == 0
        longest = 0
        if eq.size:
            for c in range(eq.shape[1]):
                cnt = mx = 0
                for v in eq[:, c]:
                    cnt = cnt + 1 if v else 0
                    mx = max(mx, cnt)
                longest = max(longest, mx)
    dt = (1.0 / nominal_fps) if nominal_fps else 0.0
    return {
        "n_joint_cols": len(cols),
        "max_abs_rad": round(max_abs, 3),
        "max_jump_rad": round(max_jump, 3),
        "longest_stuck_s": round(longest * dt, 3),
    }


def _merge_joint_obs(per_ep: list[tuple[int, dict]], qc: dict) -> dict:
    """把逐集观测量汇总成数据集级（取最坏），并附当前阈值供对照定标。"""
    if not per_ep:
        return {}
    worst_jump = max(per_ep, key=lambda t: t[1].get("max_jump_rad", 0.0))
    worst_abs = max(per_ep, key=lambda t: t[1].get("max_abs_rad", 0.0))
    worst_stuck = max(per_ep, key=lambda t: t[1].get("longest_stuck_s", 0.0))
    return {
        "n_joint_cols": per_ep[0][1].get("n_joint_cols"),
        "max_abs_rad": worst_abs[1].get("max_abs_rad"),
        "max_abs_episode": worst_abs[0],
        "max_jump_rad": worst_jump[1].get("max_jump_rad"),
        "max_jump_episode": worst_jump[0],
        "longest_stuck_s": worst_stuck[1].get("longest_stuck_s"),
        "longest_stuck_episode": worst_stuck[0],
        "thresholds": {
            "joint_limits_rad": qc.get("joint_limits_rad"),
            "joint_jump_rad": qc.get("joint_jump_rad"),
            "stuck_s": qc.get("stuck_s"),
        },
        "note": ("数组列数据无关节名（无法豁免夹爪），关节类判定默认关闭；"
                 "对照上面两个数定标后再用 --joints 打开"),
    }


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

    # 真实 v2.1 的 action / observation.state 是"每行一个数组"的 object 列：
    # 展开成 action.0…N 后 NaN/Inf 与关节类检查才真正生效（见 2026-09 审计）。
    has_arrays = dataset_io.has_array_features(eps_paths[0] if eps_paths else None)
    joints_on = bool(args.joints) or not has_arrays
    if has_arrays:
        print(f"[i] {ds.name}: 检测到数组列特征（action/observation.state 无关节名）"
              f"→ 已展开后检查 NaN/Inf；关节类判定 {'开启(--joints)' if args.joints else '默认关闭（只测量，见 qc_report）'}")

    ep_rows = []
    ep_joint_obs: list[tuple[int, dict]] = []
    meta_stats = _parse_meta_stats(ds)
    if not meta_stats:
        print(f"[i] {ds.name}: 无 meta/episodes_stats.jsonl，跳过 stats 漂移检查")
    for p in eps_paths:
        ep_idx = dataset_io.episode_index(p)
        df = dataset_io.expand_array_features(pd.read_parquet(p))
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
        q = check_episode(df, info, qc, nominal, vsum, check_joints=joints_on)
        if meta_stats.get(ep_idx):
            drift = _stats_drift(df, meta_stats[ep_idx], float(qc.get("stats_drift_tol") or 0))
            if drift:
                q["reasons_review"].append(drift)
        if has_arrays and not joints_on:
            obs = _joint_observation(df, nominal)
            if obs:
                ep_joint_obs.append((ep_idx, obs))
        if do_blur:
            b = blur_check(vsum, qc)
            if b.get("_bad"):
                q["reasons_exclude"].append(f"模糊帧占比 {b['_worst_bad_ratio']*100:.0f}% > "
                                            f"{qc['blur_bad_ratio']*100:.0f}%")
                q["verdict"] = "exclude"
        ep_rows.append({"episode": ep_idx, "n_rows": n_rows, "_qc": q, "videos": vsum})

    ep_rows.sort(key=lambda x: x["episode"])
    ds_info = _dataset_level(ep_rows, ds, qc)
    _print_counts(ep_rows)
    out_root = Path(args.out) if args.out else dataset_io.new_stage_dir(ds, "clean")
    return _finalize_qc(ds, out_root, ep_rows, nominal,
                        joint_obs=_merge_joint_obs(ep_joint_obs, qc), ds_info=ds_info)


def _print_counts(ep_rows: list[dict]) -> None:
    n_keep = sum(1 for e in ep_rows if e["_qc"]["verdict"] == "keep")
    n_rev = sum(1 for e in ep_rows if e["_qc"]["verdict"] == "review")
    n_ex = sum(1 for e in ep_rows if e["_qc"]["verdict"] == "exclude")
    print(f"      三档: keep {n_keep} / review {n_rev} / exclude {n_ex}")


# ---------------------------------------------------------------- 数据集级后处理
def _dataset_level(ep_rows: list[dict], ds: Path, qc: dict) -> dict:
    """跨集统计：维度一致性(exclude) / 时长离群(review) / stats 漂移(review) / 近重复(info)。

    在单集检查之后统一执行，并按最终原因重算 verdict（exclude > review > keep）。
    """
    info: dict[str, Any] = {}
    if not ep_rows:
        return info

    # 维度一致性：与首个集不一致 -> exclude（下游按固定维度建模型）
    dims0 = ep_rows[0]["_qc"].get("dims")
    if dims0 and any(dims0.values()):
        bad = [e["episode"] for e in ep_rows if e["_qc"].get("dims") != dims0]
        for e in ep_rows:
            if e["_qc"].get("dims") != dims0:
                e["_qc"]["reasons_exclude"].append(
                    f"特征维度与首个集不一致 {e['_qc'].get('dims')} != {dims0}")
        if bad:
            info["dim_mismatch_episodes"] = bad

    # 时长离群
    sigma = float(qc.get("duration_outlier_sigma") or 0)
    durs = np.array([e["_qc"]["duration_s"] for e in ep_rows], dtype=np.float64)
    if sigma and durs.size >= 5 and float(durs.std()) > 0:
        mu, sd = float(durs.mean()), float(durs.std())
        flagged = []
        for e in ep_rows:
            if abs(e["_qc"]["duration_s"] - mu) > sigma * sd:
                e["_qc"]["reasons_review"].append(
                    f"时长离群 {e['_qc']['duration_s']:.1f}s（均值 {mu:.1f}±{sd:.1f}s，>{sigma:g}σ）")
                flagged.append(e["episode"])
        if flagged:
            info["duration_outlier_episodes"] = flagged

    # 近重复集（指纹相同 -> info，只提示）
    fps: dict[str, list[int]] = {}
    for e in ep_rows:
        fp = e["_qc"].get("fingerprint")
        if fp:
            fps.setdefault(fp, []).append(e["episode"])
    dups = [v for v in fps.values() if len(v) > 1]
    if dups:
        info["near_duplicate_groups"] = dups

    # 统计信号汇总（info 级）：常量维 / 僵死维 / 静止帧占比 / 尖峰总数
    sig_rows = [e for e in ep_rows if e["_qc"].get("signals")]
    if sig_rows:
        n = len(sig_rows)
        agg: dict[str, Any] = {"n_episodes": n}
        for prefix in ("action", "state"):
            key = f"{prefix}_idle_ratio"
            vals = [e["_qc"]["signals"].get(key) for e in sig_rows]
            vals = [float(v) for v in vals if v is not None]
            if vals:
                agg[key] = {"p50": round(float(np.median(vals)), 3),
                            "p90": round(float(np.percentile(vals, 90)), 3),
                            "max": round(max(vals), 3)}
            for name, field in (("stuck", f"{prefix}_stuck_dims"),
                                ("const", f"{prefix}_const_dims")):
                cnt: dict[int, int] = {}
                for e in sig_rows:
                    for d in e["_qc"]["signals"].get(field) or []:
                        cnt[int(d)] = cnt.get(int(d), 0) + 1
                if cnt:
                    agg[f"{prefix}_{name}_dims"] = {
                        f"dim{d}": f"{c}/{n} 集"
                        for d, c in sorted(cnt.items(), key=lambda kv: -kv[1])[:12]}
            jumps = [(e["episode"], e["_qc"]["signals"].get(f"{prefix}_max_jump")) for e in sig_rows]
            jumps = [(ep, float(v)) for ep, v in jumps if v is not None]
            jumps.sort(key=lambda t: -t[1])
            if jumps:
                agg[f"{prefix}_max_jump_top"] = {f"ep{ep}": v for ep, v in jumps[:3]}
            agg[f"{prefix}_spikes_total"] = int(sum(
                int(e["_qc"]["signals"].get(f"{prefix}_spikes") or 0) for e in sig_rows))
        info["signals_summary"] = agg

    for e in ep_rows:
        q = e["_qc"]
        q["verdict"] = ("exclude" if q["reasons_exclude"]
                        else ("review" if q["reasons_review"] else "keep"))
    return info


def _finalize_qc(ds: Path, out_root: Path, ep_rows: list[dict], nominal: float,
                 joint_obs: dict | None = None, ds_info: dict | None = None) -> dict:
    """写 episode_disposition.csv / summary.json / qc_report.md（v2.1/v3.0 共用）。

    verdict 三档：exclude（硬伤，05 合并排除）/ review（需人看，05 不排除）/
    keep。review 与 info 只提示，绝不删数据。
    """
    out_root.mkdir(parents=True, exist_ok=True)
    n_keep = sum(1 for e in ep_rows if e["_qc"]["verdict"] == "keep")
    n_review = sum(1 for e in ep_rows if e["_qc"]["verdict"] == "review")
    n_excl = sum(1 for e in ep_rows if e["_qc"]["verdict"] == "exclude")

    rows = []
    for e in ep_rows:
        r = {"episode": e["episode"], "n_rows": e["n_rows"], "verdict": e["_qc"]["verdict"],
             "duration_s": e["_qc"]["duration_s"],
             "reasons_exclude": " | ".join(e["_qc"]["reasons_exclude"]) or "-",
             "reasons_review": " | ".join(e["_qc"].get("reasons_review") or []) or "-",
             "reasons_warn": " | ".join(e["_qc"]["reasons_warn"]) or "-"}
        for cam, v in e["videos"].items():
            r[f"video_{cam}"] = ("缺失" if v["missing"] else
                                 (f"{v['frames']}" if v["frames"] is not None else "n/a"))
        rows.append(r)
    report.write_csv(out_root / "episode_disposition.csv", rows)

    review_eps = [e["episode"] for e in ep_rows if e["_qc"]["verdict"] == "review"]
    summary = {
        "dataset": ds.name, "path": str(ds), "nominal_fps": nominal,
        "n_episodes": len(ep_rows), "n_keep": n_keep, "n_review": n_review, "n_exclude": n_excl,
        "excluded_episodes": [e["episode"] for e in ep_rows if e["_qc"]["verdict"] == "exclude"],
        "review_episodes": review_eps,
        "disposition_csv": str(out_root / "episode_disposition.csv"),
    }
    if joint_obs:
        summary["joint_observation"] = joint_obs
    if ds_info:
        summary["dataset_level"] = ds_info
    report.write_json(out_root / "summary.json", summary)

    md = [
        f"# 质检报告: {ds.name}", "",
        f"- keep {n_keep} / **review {n_review}** / exclude {n_excl}（共 {len(ep_rows)} 集）",
        f"- 排除集（05 合并会剔除）: {summary['excluded_episodes'] or '无'}",
        f"- 待复核集（05 不剔除，建议人看/进盲审页）: {review_eps[:30] or '无'}"
        f"{' …' if len(review_eps) > 30 else ''}",
        "", "## 逐集", "",
        "| ep | 行数 | 时长s | 结论 | 原因 |", "|---|---|---|---|---|",
    ]
    for e in ep_rows:
        q = e["_qc"]
        why = "; ".join(q["reasons_exclude"]
                        + [f"🔍 {r}" for r in (q.get("reasons_review") or [])]
                        + [f"⚠ {w}" for w in q["reasons_warn"]]) or "✓"
        md.append(f"| {e['episode']} | {e['n_rows']} | {q['duration_s']:.2f} | {q['verdict']} | {why} |")
    if ds_info:
        bits = []
        if ds_info.get("dim_mismatch_episodes"):
            bits.append(f"- 维度不一致（已 exclude）: {ds_info['dim_mismatch_episodes']}")
        if ds_info.get("duration_outlier_episodes"):
            bits.append(f"- 时长离群（review）: {ds_info['duration_outlier_episodes']}")
        if ds_info.get("near_duplicate_groups"):
            bits.append(f"- 近重复集分组（info）: {ds_info['near_duplicate_groups']}")
        if bits:
            md += ["", "## 跨集统计", "", *bits]
        agg = ds_info.get("signals_summary")
        if agg:
            md += ["", "## 统计信号（数据级汇总，info 级）", "",
                   "常量维 = 该维标准差≈0（通道没数据）；僵死维 = 该维多数帧不变（夹爪常见）；"
                   "静止帧占比 = 全维位移小于阈值的帧比例。", "",
                   "| 指标 | 值 |", "|---|---|"]
            for k, v in agg.items():
                if isinstance(v, dict):
                    v = " / ".join(f"{kk}:{vv}" for kk, vv in v.items())
                md.append(f"| {k} | {v} |")
    if joint_obs:
        th = joint_obs.get("thresholds", {})
        md += [
            "", "## 关节观测量（只测量，未参与判定）", "",
            "数组列数据没有关节名、无法豁免夹爪维度，故关节类判定默认关闭。"
            "先看下面的实测值，再决定 `qc.*` 阈值并用 `--joints` 打开判定。", "",
            "| 观测量 | 实测(全数据集最坏) | 出现在 | 当前阈值 |", "|---|---|---|---|",
            f"| max 关节角绝对值 (rad) | {joint_obs.get('max_abs_rad')} | ep {joint_obs.get('max_abs_episode')} "
            f"| joint_limits_rad={th.get('joint_limits_rad')} |",
            f"| max 相邻帧跳变 (rad) | {joint_obs.get('max_jump_rad')} | ep {joint_obs.get('max_jump_episode')} "
            f"| joint_jump_rad={th.get('joint_jump_rad')} |",
            f"| 最长零方差持续 (s) | {joint_obs.get('longest_stuck_s')} | ep {joint_obs.get('longest_stuck_episode')} "
            f"| stuck_s={th.get('stuck_s')} |",
            f"| 关节维度数 | {joint_obs.get('n_joint_cols')} | — | — |",
            "",
        ]
    md += ["", "## 说明", "", "- 软标记：本脚本未修改任何数据文件；05 合并时按 episode_disposition.csv 排除 exclude 集。",
           "- 数组列（action/observation.state）已展开后检查 NaN/Inf；v2.1 与 v3.0 行为一致。", ""]
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
    joints_on = bool(args.joints)
    print(f"[INFO] {ds.name} 为 v3.0，数组列已展开；关节类判定 "
          f"{'开启(--joints)' if joints_on else '默认关闭（只测量，见 qc_report）'}")
    vid_by_ep = (dataset_io._v3_episode_video_summary(ds, eps_meta, cams)
                 if do_videos else {})
    ep_rows = []
    ep_joint_obs: list[tuple[int, dict]] = []
    meta_stats = _parse_meta_stats(ds)
    for ep, df in dataset_io.iter_v3_episodes(ds):
        vsum = vid_by_ep.get(ep, {})
        q = check_episode(df, info, qc, nominal, vsum, check_joints=joints_on)
        if meta_stats.get(ep):
            drift = _stats_drift(df, meta_stats[ep], float(qc.get("stats_drift_tol") or 0))
            if drift:
                q["reasons_review"].append(drift)
        if not joints_on:
            obs = _joint_observation(df, nominal)
            if obs:
                ep_joint_obs.append((ep, obs))
        ep_rows.append({"episode": ep, "n_rows": len(df), "_qc": q, "videos": vsum})

    ep_rows.sort(key=lambda x: x["episode"])
    ds_info = _dataset_level(ep_rows, ds, qc)
    _print_counts(ep_rows)
    out_root = Path(args.out) if args.out else dataset_io.new_stage_dir(ds, "clean")
    return _finalize_qc(ds, out_root, ep_rows, nominal,
                        joint_obs=_merge_joint_obs(ep_joint_obs, qc), ds_info=ds_info)


def main() -> int:
    ap = argparse.ArgumentParser(description="LeRobot v2.1 清洗/质检（软标记，只读）")
    ap.add_argument("--input", action="append", required=True)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--no-video-check", action="store_true")
    ap.add_argument("--joints", action="store_true",
                    help="数组列数据（无关节名）也启用关节限位/跳变/卡死判定。"
                         "默认关闭：先用报告里的『关节观测量』定标 qc.* 阈值，再打开")
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