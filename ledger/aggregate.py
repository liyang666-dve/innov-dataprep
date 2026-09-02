#!/usr/bin/env python3
"""台账汇总：把各台机器的 data_catalog.csv 合并成总台账（第四台电脑上跑）。

去重口径（对应入库SOP）：batch_id 相同只保留第一条，重复的计数报告。

用法:
    python3 ledger/aggregate.py --dir <台账目录>          # 扫描目录下所有 data_catalog*.csv
    python3 ledger/aggregate.py --inputs a.csv b.csv      # 或显式指定文件
    python3 ledger/aggregate.py --dir <目录> --out merged.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    ap = argparse.ArgumentParser(description="合并多机台账")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dir", help="扫描该目录下 data_catalog*.csv")
    g.add_argument("--inputs", nargs="+", help="显式 csv 文件列表")
    ap.add_argument("--out", default=None, help="输出（默认 <dir>/data_catalog_merged.csv）")
    args = ap.parse_args()

    files: list[Path] = []
    if args.dir:
        d = Path(args.dir)
        files = sorted(d.glob("data_catalog*.csv"))
        if not files:
            print(f"[!] {d} 下没有 data_catalog*.csv")
            return 1
    else:
        files = [Path(p) for p in args.inputs]

    rows: list[dict] = []
    seen: set[str] = set()
    dup = 0
    for f in files:
        for r in read_csv(f):
            if not r or not r.get("batch_id"):
                continue
            if r["batch_id"] in seen:
                dup += 1
                continue
            seen.add(r["batch_id"])
            r["_source"] = f.name
            rows.append(r)

    out = Path(args.out) if args.out else (Path(args.dir) / "data_catalog_merged.csv" if args.dir else Path("data_catalog_merged.csv"))
    cols = list(rows[0].keys()) if rows else []
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"来源文件: {len(files)} 个")
    print(f"总记录: {len(rows)} 条（去重后），重复跳过: {dup} 条")
    print(f"输出: {out}")
    if rows:
        for r in rows[:20]:
            print(f"  {r.get('batch_id')} | {r.get('task')} | {r.get('date')} | "
                  f"{r.get('episodes')}集 | {r.get('fps')}fps | {r.get('duration_h')}h | "
                  f"质量:{r.get('quality')} | 机器:{r.get('_source')}")
        if len(rows) > 20:
            print(f"  ... 共 {len(rows)} 条")
    return 0


if __name__ == "__main__":
    sys.exit(main())