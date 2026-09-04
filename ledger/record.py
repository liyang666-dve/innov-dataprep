#!/usr/bin/env python3
"""登记卡：数据处理达标后（默认 final 阶段）生成登记卡并写入台账 data_catalog.csv。

时机约定（见 README"登记"一节）：
  - --stage final（默认）：在 05 合并 / 06 转换之后对"最终数据集"登记，质量默认 clean；
  - --stage raw（可选）：对每个原始采集批次登记，质量默认 raw（保留"每批一行"粒度）。

自动推导：批次号(命名公式) / 任务 / 日期 / 机型 / 相机 / 集数 / 总帧数 / 帧率 /
总时长 / 平均时长 / 格式 / 质量。
人工确认：操作员 / 采集机 / 备注（--operator / --machine / --note 或交互输入）。

防呆：
  - 非 v2.1 / 空数据集（0 集）→ [ERROR] 拒绝登记，绝不写垃圾行；
  - 日期解析不出且无时间戳 → [ERROR] 提示用 --date；
  - batch_id 重复 → 拦截（--force 除外）。

用法:
    python3 ledger/record.py --batch <数据集目录> [--config config.yaml] [--yes]
          [--stage final|raw] [--operator 张三] [--note '合并自 0901、0902'] [--date 0730-0731]
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
    "sensors", "format", "quality", "stage", "source", "stats", "note", "registered_at",
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


def sensors_str(cameras: dict) -> str:
    parts = []
    for k, v in cameras.items():
        w, h = v.get("w") or 0, v.get("h") or 0
        parts.append(f"{k}({w}x{h})" if w and h else k)
    return ", ".join(parts) or "-"


def pick_task(summary: dict, args: argparse.Namespace, cfg: dict) -> str:
    """任务取值链：--task > defaults.task > meta 单任务 > [ERROR]。"""
    defaults = cfg.get("defaults", {})
    task_names = summary["task_names"]
    explicit = args.task or defaults.get("task") or ""
    if explicit:
        return sanitize_task(explicit)
    if len(task_names) == 1:
        return sanitize_task(task_names[0])
    if len(task_names) > 1:
        print(f"[ERROR] 数据含多个任务（{task_names}），请用 --task 或 config.defaults.task 指定")
        sys.exit(2)
    print("[ERROR] 无法确定任务：--task / config.defaults.task / meta 任务 都没有")
    sys.exit(2)


def pick_date(summary: dict, args: argparse.Namespace) -> tuple[str, str]:
    """日期：--date > 时间戳 > 目录名解析。解析不出 -> [ERROR]。"""
    if args.date:
        m = re.fullmatch(r"(\d{4})-(\d{4})|(\d{4})", args.date)
        if not m:
            print(f"[ERROR] --date 格式应为 MMDD 或 MMDD-MMDD，收到: {args.date}")
            sys.exit(2)
        a = m.group(1) or m.group(3)
        b = m.group(2) or a
        return a, b
    if summary.get("min_date"):
        return mmdd(summary["min_date"], "????"), mmdd(summary["max_date"], "????")
    parsed = dataset_io.parse_date_range(summary.get("name", ""))
    if parsed:
        return parsed
    print("[ERROR] 无法从目录名解析日期且数据无时间戳，请用 --date 0730 或 0730-0731 指定")
    sys.exit(2)


def build_record(summary: dict, cfg: dict, args: argparse.Namespace) -> dict:
    defaults = cfg.get("defaults", {})
    robot_map = cfg.get("robot_type_map", {})

    task = pick_task(summary, args, cfg)
    robot = robot_map.get(summary["robot_type"], "unk")
    if robot == "unk":
        print(f"[!] robot_type '{summary['robot_type']}' 未在 config.robot_type_map 中，登记为 unk")

    ncam = len(summary["cameras"])
    start_m, end_m = pick_date(summary, args)
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

    stage = args.stage
    quality = args.quality or ("clean" if stage == "final" else "raw")

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
        "quality": quality,
        "stage": stage,
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
        kv("阶段/质量", f"stage={r['stage']} / {r['quality']} / {r['format']}"),
        kv("来源 source", r["source"]),
        kv("备注 note", r["note"] or "(空)"),
        "-" * 56,
        "提示：以上除 操作员/采集机/备注 外均为自动推导，直接确认即可。",
        "=" * 56,
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


def _main() -> int:
    ap = argparse.ArgumentParser(description="生成登记卡并写入台账（默认 final 阶段）")
    ap.add_argument("--batch", required=True, help="数据集目录（合并/转换后的最终数据集）")
    ap.add_argument("--config", default="config.yaml", help="配置文件（默认 ./config.yaml，可不存在）")
    ap.add_argument("--stage", choices=["final", "raw"], default="final",
                    help="final=处理达标后登记（默认，质量 clean）；raw=原始批次登记（质量 raw）")
    ap.add_argument("--yes", action="store_true", help="跳过确认直接写入")
    ap.add_argument("--force", action="store_true", help="批次号已存在时仍写入")
    ap.add_argument("--task", default=None)
    ap.add_argument("--operator", default=None)
    ap.add_argument("--machine", default=None)
    ap.add_argument("--version", default=None)
    ap.add_argument("--quality", default=None, help="覆盖质量（默认 final->clean / raw->raw）")
    ap.add_argument("--date", default=None, help="日期覆盖，MMDD 或 MMDD-MMDD")
    ap.add_argument("--note", default=None)
    ap.add_argument("--out", default=None, help="台账 csv 路径（默认 config.paths.ledger 或 ./data_catalog.csv）")
    args = ap.parse_args()

    ds = Path(args.batch)
    kind, reason = dataset_io.detect_dataset(ds)
    if kind not in ("v2.1", "v3.0"):
        print(f"[ERROR] 拒绝登记：{ds.name} 不是 v2.1/v3.0 数据集 -> {reason}")
        return 1

    cfg = load_config(Path(args.config))
    ledger = Path(args.out or cfg.get("paths", {}).get("ledger") or "data_catalog.csv")

    summary = dataset_io.summarize_dataset(ds, check_videos=False)
    if summary["n_episodes"] == 0:
        print("[ERROR] 拒绝登记：0 个 episode（空数据集？），不写入台账")
        return 1

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


def main() -> int:
    try:
        return _main()
    except OSError as e:
        print(f"[ERROR] 写入台账失败（路径不可写？）: {e}")
        print("        可换目录重试: python3 ledger/record.py --out ./data_catalog.csv ...")
        return 1


if __name__ == "__main__":
    sys.exit(main())