#!/usr/bin/env python3
"""02 时间戳审计：单调性 / 重复 / 丢帧窗口（只读，不改数据）。

设计说明：
- LeRobot v2.1 的 parquet 行与 mp4 帧按 index 一一对应，**删除行会破坏视频对齐**，
  因此本脚本只做审计并输出建议，真正的"剔除/修复"由后续 03 清洗（软标记）处理。
- 丢帧窗口：帧间隔 > 2×中位数 视为一次丢帧，估计丢失帧数。

用法:
    python3 pipe/02_timestamps.py --input <dataset> [--input ...] [--out <目录>]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipe.lib import dataset_io, report  # noqa: E402


def dropped_windows(df: pd.DataFrame) -> list[dict[str, float]]:
    """返回时间戳中的丢帧窗口列表。"""
    if "timestamp" not in df.columns:
        return []
    t = df["timestamp"].to_numpy(dtype=np.float64)
    if np.isfinite(t).sum() < 2:
        return []
    t = t[np.isfinite(t)]
    d = np.diff(t)
    med = float(np.median(d))
    if med <= 0:
        return []
    wins: list[dict[str, float]] = []
    i = 0
    while i < len(d):
        if d[i] > 2 * med:
            j = i
            while j + 1 < len(d) and d[j + 1] > 2 * med:
                j += 1
            gap = float(d[i:j + 1].sum())
            wins.append({
                "from_idx": int(i), "to_idx": int(j + 1),
                "duration_s": round(gap, 4),
                "dropped_est": int(round(gap / med) - 1),
                "gap_s": round(gap, 4),
            })
            i = j + 1
        else:
            i += 1
    return wins


def run_dataset(ds: Path, out_root: Path) -> tuple[int, list[dict]]:
    """返回 (问题数, 窗口行)。"""
    fps = 30.0
    eps_paths = dataset_io.discover_episodes(ds)
    rows: list[dict] = []
    all_windows: list[dict] = []
    n_problems = 0
    for p in eps_paths:
        df = pd.read_parquet(p)
        n = len(df)
        wins = dropped_windows(df)
        for w in wins:
            w2 = {"dataset": ds.name, "episode": dataset_io.episode_index(p), **w}
            all_windows.append(w2)
        row = {
            "episode": dataset_io.episode_index(p),
            "n_rows": n,
            "n_duplicate_ts": int((df["timestamp"].diff() <= 0).sum()) if "timestamp" in df.columns else 0,
            "n_backward_ts": int((df["timestamp"].diff() < 0).sum()) if "timestamp" in df.columns else 0,
            "n_windows": len(wins),
            "dropped_est_total": sum(w["dropped_est"] for w in wins),
            "max_gap_s": round(max((w["gap_s"] for w in wins), default=0.0), 4),
        }
        if "timestamp" in df.columns:
            t = df["timestamp"].to_numpy(dtype=np.float64)
            t = t[np.isfinite(t)]
            if t.size >= 2:
                d = np.diff(t)
                med = float(np.median(d))
                row["median_dt"] = round(med, 6)
                row["actual_fps"] = round((n - 1) / float(t[-1] - t[0]), 3) if t[-1] > t[0] else None
        rows.append(row)
        if row["n_windows"] > 0 or row["n_duplicate_ts"] > 0:
            n_problems += 1

    out_root.mkdir(parents=True, exist_ok=True)
    report.write_csv(out_root / "timestamps_summary.csv", rows)
    report.write_csv(out_root / "dropped_windows.csv", all_windows)

    lines = [
        f"# 时间戳审计: {ds.name}",
        "",
        f"- 问题集数: {n_problems} / {len(rows)}",
        f"- 丢帧窗口总数: {len(all_windows)}，累计丢失估计: {sum(w['dropped_est'] for w in all_windows)} 帧",
        "",
        "## 逐集",
        "",
        "| ep | 行数 | 重复戳 | 回退 | 窗口数 | 丢帧估计 | 最大间隔s | 中位dt | 实际fps |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['episode']} | {r['n_rows']} | {r['n_duplicate_ts']} | {r['n_backward_ts']} "
                     f"| {r['n_windows']} | {r['dropped_est_total']} | {r['max_gap_s']} "
                     f"| {report.fmt_x(r.get('median_dt'), 6)} | {report.fmt_x(r.get('actual_fps'), 1)} |")
    lines += [
        "",
        "## 建议",
        "",
        "- 丢帧/重复戳的集：**不要直接删 parquet 行**（会与 mp4 帧错位），等 03 清洗按集软标记或重采；",
        "- 单集丢帧估计 > 5% 且任务关键：建议重采该集；",
        "- 所有集都系统性丢帧：先查采集链路（USB 带宽 / CPU 抢占 / CAN 掉线），再批量重采。",
        "",
    ]
    report.write_md(out_root / "timestamps_report.md", report_lines := lines)
    return n_problems, all_windows


def main() -> int:
    ap = argparse.ArgumentParser(description="LeRobot v2.1 时间戳审计（只读）")
    ap.add_argument("--input", action="append", required=True, help="数据集目录（可多次）")
    ap.add_argument("--out", default=None, help="输出目录（默认 <输入父目录>/<名字>_timestamps/）")
    args = ap.parse_args()
    rc = 0
    for ds_str in args.input:
        ds = Path(ds_str)
        if not dataset_io.is_dataset_dir(ds):
            print(f"[跳过] 不是 v2.1 数据集: {ds}")
            rc = 1
            continue
        out = Path(args.out) if args.out else ds.parent / f"{ds.name}_timestamps"
        n, wins = run_dataset(ds, out)
        print(f"[OK] {ds.name}: 问题集 {n}，丢帧窗口 {len(wins)} -> {out / 'timestamps_report.md'}")
    return rc


if __name__ == "__main__":
    sys.exit(main())