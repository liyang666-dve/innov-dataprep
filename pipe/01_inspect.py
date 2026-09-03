#!/usr/bin/env python3
"""01 盘点：对 LeRobot v2.1 数据集输出盘点报告（只读，不修改数据）。

逐集统计：帧数 / 时长 / 实际帧率(与标称对比) / 时间戳(重复/回退/丢帧估计) /
NaN / 视频帧数与 parquet 行数对齐情况。

用法:
    python3 pipe/01_inspect.py --input <dataset> [--input <dataset2> ...]
          [--fps 30] [--no-video-check] [--out <报告目录>]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipe.lib import dataset_io, report, video_utils  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LeRobot v2.1 数据集盘点（只读）")
    p.add_argument("--input", action="append", required=True, help="数据集目录（可多次）")
    p.add_argument("--fps", type=float, default=None, help="标称帧率，覆盖 info.json（默认用 info.json）")
    p.add_argument("--no-video-check", action="store_true", help="跳过视频帧数核对（大集省时间）")
    p.add_argument("--out", default=None, help="报告输出目录（默认 <输入父目录>/<名字>_inspect/）")
    return p.parse_args()


def build_csv_rows(summary: dict) -> list[dict]:
    rows = []
    for e in summary["episodes"]:
        r = {
            "episode": e["episode"],
            "n_rows": e["n_rows"],
            "duration_s": e.get("duration_s"),
            "fps_actual": e.get("fps_actual"),
            "fps_nominal": summary["fps_nominal"],
            "n_dup_ts": e.get("n_dup_ts", 0),
            "n_backward_ts": e.get("n_backward_ts", 0),
            "max_dt": e.get("max_dt"),
            "median_dt": e.get("median_dt"),
            "dropped_est": e.get("dropped_est", 0),
            "n_nan": e.get("n_nan", 0),
            "nan_cols": ",".join(e.get("nan_cols", [])),
        }
        for cam, v in e.get("videos", {}).items():
            r[f"video_{cam}_frames"] = v["frames"]
            r[f"video_{cam}_mismatch"] = v["mismatch"]
            r[f"video_{cam}_missing"] = v["missing"]
        rows.append(r)
    return rows


def build_md(summary: dict) -> list[str]:
    L: list[str] = []
    L.append(f"# 盘点: {summary['name']}")
    L.append("")
    L.append("## 元信息")
    L.append("")
    L.append("| 项 | 值 |")
    L.append("|---|---|")
    L.append(f"| 路径 | `{summary['path']}` |")
    L.append(f"| 格式 | {summary['format']} |")
    L.append(f"| robot_type | {summary['robot_type']} |")
    L.append(f"| 标称帧率 | {summary['fps_nominal']} |")
    L.append(f"| 任务 | {', '.join(summary['task_names']) or '-'} |")
    cam_str = ", ".join(f"{k}({v.get('w') or 0}x{v.get('h') or 0})" for k, v in summary["cameras"].items()) or "-"
    L.append(f"| 相机 | {cam_str} |")
    L.append(f"| 集数 | {summary['n_episodes']} |")
    L.append(f"| 总帧数 | {summary['total_frames']} |")
    L.append(f"| 总时长 | {summary['duration_h']} h |")
    L.append(f"| 时间范围 | {summary.get('min_date') or '-'} ~ {summary.get('max_date') or '-'} |")
    L.append("")
    L.append("## 逐集")
    L.append("")
    L.append("| ep | 帧数 | 时长s | 实际fps | 重复戳 | 回退 | 丢帧估计 | NaN | 视频对齐 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for e in summary["episodes"]:
        vags = []
        for cam, v in e.get("videos", {}).items():
            if v["missing"]:
                vags.append(f"{cam}:缺失")
            elif v["frames"] is None:
                vags.append(f"{cam}:n/a")
            elif v["mismatch"]:
                vags.append(f"{cam}:{v['mismatch']:+d}")
            else:
                vags.append(f"{cam}:✓")
        L.append(f"| {e['episode']} | {e['n_rows']} | {report.fmt_x(e.get('duration_s'), 2)} "
                 f"| {report.fmt_x(e.get('fps_actual'), 1)} | {e.get('n_dup_ts', 0)} "
                 f"| {e.get('n_backward_ts', 0)} | {e.get('dropped_est', 0)} | {e.get('n_nan', 0)} "
                 f"| {', '.join(vags) or '-'} |")
    L.append("")
    L.append("## 问题清单")
    L.append("")
    if summary["issues"]:
        for it in summary["issues"]:
            L.append(f"- {it}")
    else:
        L.append("- 未发现问题")
    L.append("")
    return L


def main() -> int:
    args = parse_args()
    rc = 0
    for ds_str in args.input:
        ds = Path(ds_str)
        kind, reason = dataset_io.detect_dataset(ds)
        if kind != "v2.1":
            print(f"[跳过] {ds.name} 不是 v2.1 数据集: {reason}")
            rc = 1
            continue
        fps = args.fps if args.fps else None
        summary = dataset_io.summarize_dataset(ds, fps_nominal=fps or 30.0,
                                               check_videos=not args.no_video_check)
        if not args.no_video_check and video_utils.FFPROBE is None:
            print(f"[WARN] ffprobe 不可用，{ds.name} 的视频帧数核对已跳过（sudo apt install ffmpeg）")
        out_dir = Path(args.out) if args.out else ds.parent / f"{ds.name}_inspect"
        report.write_csv(out_dir / "episodes_summary.csv", build_csv_rows(summary))
        report.write_json(out_dir / "summary.json", summary)
        report.write_md(out_dir / "report.md", build_md(summary))
        print(f"[OK] {ds.name}: {summary['n_episodes']} 集 / {summary['total_frames']} 帧 / "
              f"{summary['duration_h']}h / {len(summary['issues'])} 个问题 -> {out_dir / 'report.md'}")
        for it in summary["issues"][:15]:
            print(f"      ! {it}")
        if len(summary["issues"]) > 15:
            print(f"      ... 共 {len(summary['issues'])} 个问题，详见 report.md")
    return rc


if __name__ == "__main__":
    sys.exit(main())