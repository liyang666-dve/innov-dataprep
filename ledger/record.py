#!/usr/bin/env python3
"""登记卡：每采集/处理完一批后，生成登记卡并写入台账 data_catalog.csv。

自动推导：批次号(命名公式) / 任务 / 日期 / 机型 / 相机 / 集数 / 总帧数 / 帧率 /
总时长 / 平均时长 / 格式 / 质量（默认 raw）。
人工确认：操作员 / 采集机 / 备注（--operator / --machine / --note 或交互输入）。

用法:
    python3 ledger/record.py --batch <数据集目录> [--config config.yaml] [--yes]
          [--operator 张三] [--note '合并自 0901、0902 两批'] [--quality raw]
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipe.lib import dataset_io  # noqa: E402

LEDGER_COLUMNS = [
    "batch_id", "task", "date", "total_days", "robot", "machine", "operator", "version",
    "episodes", "total_frames", "fps", "duration_h", "avg_duration_min",
    "sensors", "format", "quality", "source", "stats", "note", "registered_at",
]

DATE_PARSERS = [
    lambda s: re.search(r"(\d{8})-(\d{8})", s),   # 20260730-20260731
    lambda s: re.search(r"(\d{4})-(\d{4})", s),   # 0730-0731
    lambda s: re.search(r"(\d{4})", s),           # 0730 / 0901
]


def load_config(path: Path | None) -> dict:
    if not path or not Path(path).is_file():
        return {}
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def sanitize_task(t: str | None) -> str:
    t = (t or "").strip()
    t = re.sub(r"\s+", "_", t)
    return t or "task"


def mmdd(iso: str | None, default: str) -> str:
    if not iso:
        return default
    try:
        return date.fromisoformat(iso).strftime("%m%d")
    except ValueError:
        return default


def num_days(start_iso: str | None, end_iso: str | None) -> int | str:
    try:
        if start_iso and end_iso:
            return max(1, (date.fromisoformat(end_iso) - date.fromisoformat(start_iso)).days + 1)
    except ValueError:
        pass
    return "?"


def parse_dir_date(name: str) -> tuple[str, str]:
    """目录名日期回退：返回 (start_MMDD, end_MMDD)。"""
    for p in DATE_PARSERS:
        m = p(name)
        if m:
            part = m.group(1).replace("-", "")
            a, b = part[:4], part[4:]
            return a, b
    return "????", "????"


def sensors_str(cameras: dict) -> str:
    parts = []
    for k, v in cameras.items():
        w, h = v.get("w") or 0, v.get("h") or 0
        parts.append(f"{k}({w}x{h})" if w and h else k)
    return ", ".join(parts) or "-"


def build_record(summary: dict, cfg: dict, args: argparse.Namespace) -> dict:
    defaults = cfg.get("defaults", {})
    robot_map = cfg.get("robot_type_map", {})

    task = sanitize_task(args.task or (summary["task_names"][0] if len(summary["task_names"]) == 1 else None))
    if not args.task and len(summary["task_names"]) != 1:
        print(f"[!] meta 中有 {len(summary['task_names'])} 个任务，请用 --task 指定任务名")
        print(f"    可用任务: {summary['task_names']}")
        sys.exit(2)

    robot = robot_map.get(summary["robot_type"], "unk")
    if robot == "unk":
        print(f"[!] robot_type '{summary['robot_type']}' 未在 config.robot_type_map 中，登记为 unk")

    ncam = len(summary["cameras"])
    start_m, end_m = parse_dir_date(summary["name"])
    if summary.get("min_date"):
        start_iso = summary["min_date"][:10]
        end_iso = summary["max_date"][:10]
        start_m = mmdd(start_iso, start_m)
        end_m = mmdd(end_iso, end_m)
    date_str = f"{start_m}-{end_m}" if start_m != end_m else start_m

    version = args.version or defaults.get("version", "v1")
    m = re.search(r"_v(\d+)", summary["name"])
    if m and not args.version:
        version = f"v{m.group(1)}"

    batch_id = f"{task}_{robot}_{date_str}_{ncam}cam_{version}"
    fps = summary["fps_nominal"]
    episodes = summary["n_episodes"]
    hours = summary["duration_h"]
    avg_min = round(hours * 60 / episodes, 2) if episodes else 0.0
    stats = f"{episodes}集/{summary['total_frames']}帧/{fps}fps/{hours}h"

    return {
        "batch_id": batch_id,
        "task": task,
        "date": date_str,
        "total_days": num_days(summary.get("min_date"), summary.get("max_date")),
        "robot": robot,
        "machine": args.machine or defaults.get("machine", "采集机A"),
        "operator": args.operator or defaults.get("operator", ""),
        "version": version,
        "episodes": episodes,
        "total_frames": summary["total_frames"],
        "fps": fps,
        "duration_h": hours,
        "avg_duration_min": avg_min,
        "sensors": sensors_str(summary["cameras"]),
        "format": summary["format"],
        "quality": args.quality or "raw",
        "source": summary["path"],
        "stats": stats,
        "note": args.note or "",
        "registered_at": datetime.now().isoformat(timespec="seconds"),
    }


def print_card(r: dict) -> None:
    w = 14
    def kv(k: str, v, unit: str = "") -> str:
        return f"{k:<{w}}: {v}{unit}"
    lines = [
        "=" * 56,
        kv("批次号 batch_id", r["batch_id"]),
        kv("任务 task", r["task"]),
        kv("采集日期 date", r["date"], f"  (总天数 {r['total_days']})"),
        kv("机型 robot", r["robot"], f"   | 采集机: {r['machine']}"),
        kv("操作员 operator", r["operator"] or "(空)"),
        kv("数据版本 version", r["version"]),
        kv("集数/帧数", f"{r['episodes']} / {r['total_frames']}", f"  | {r['fps']}fps / {r['duration_h']}h"),
        kv("平均时长 avg", f"{r['avg_duration_min']} min/集"),
        kv("传感器 sensors", r["sensors"]),
        kv("格式/质量", f"{r['format']} / {r['quality']}"),
        kv("来源 source", r["source"]),
        kv("备注 note", r["note"] or "(空)"),
        "-" * 56,
    ]
    print("\n".join(lines))


def read_existing_batch_ids(ledger: Path) -> set[str]:
    if not ledger.is_file():
        return set()
    with open(ledger, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if "batch_id" not in (reader.fieldnames or []):
            return set()
        return {row["batch_id"] for row in reader if row.get("batch_id")}


def append_row(ledger: Path, rec: dict) -> None:
    ledger.parent.mkdir(parents=True, exist_ok=True)
    new = not ledger.exists()
    with open(ledger, "a", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_COLUMNS)
        if new:
            w.writeheader()
        w.writerow(rec)


def main() -> int:
    ap = argparse.ArgumentParser(description="生成登记卡并写入台账")
    ap.add_argument("--batch", required=True, help="数据集目录（本批次）")
    ap.add_argument("--config", default="config.yaml", help="配置文件（默认 ./config.yaml，可不存在）")
    ap.add_argument("--yes", action="store_true", help="跳过确认直接写入")
    ap.add_argument("--force", action="store_true", help="批次号已存在时仍写入")
    ap.add_argument("--task", default=None)
    ap.add_argument("--operator", default=None)
    ap.add_argument("--machine", default=None)
    ap.add_argument("--version", default=None)
    ap.add_argument("--quality", default=None, help="raw/clean")
    ap.add_argument("--note", default=None)
    ap.add_argument("--out", default=None, help="台账 csv 路径（默认 config.paths.ledger 或 ./data_catalog.csv）")
    args = ap.parse_args()

    ds = Path(args.batch)
    if not dataset_io.is_dataset_dir(ds):
        print(f"[错误] 不是 v2.1 数据集: {ds}")
        return 1

    cfg = load_config(Path(args.config))
    ledger = Path(args.out or cfg.get("paths", {}).get("ledger") or "data_catalog.csv")

    summary = dataset_io.summarize_dataset(ds, check_videos=False)
    rec = build_record(summary, cfg, args)
    print_card(rec)

    existing = read_existing_batch_ids(ledger)
    if rec["batch_id"] in existing and not args.force:
        print(f"[!] batch_id '{rec['batch_id']}' 已存在，跳过（--force 可强制再写）")
        return 1

    if not args.yes:
        ans = input("确认写入台账? [y/N] ").strip().lower()
        if ans not in ("y", "yes", "是"):
            print("已取消，未写入。")
            return 0

    append_row(ledger, rec)
    print(f"[OK] 已登记 -> {ledger}")
    return 0


if __name__ == "__main__":
    sys.exit(main())