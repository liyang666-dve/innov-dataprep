#!/usr/bin/env python3
"""00_ingest —— INGEST 适配入口：任一采集源原始目录 → 标准 LeRobot v2.1 数据集。

用法:
    python3 pipe/00_ingest.py --source umi  --input <UMI原始目录> [--output <根>] [--task xxx] [--robot xxx] [--fps 30]
    python3 pipe/00_ingest.py --source ego  --input <ego原始目录>
    python3 pipe/00_ingest.py --source sim  --input <sim导出目录>
    python3 pipe/00_ingest.py --source teleop --input <已是 v2.1 的目录>   # 直通+补 source_meta
    python3 pipe/00_ingest.py --source umi --demo      # 一键生成 UMI 合成样例并转换（家里自测/演示）

输出：<output>/<task>_<robot>_<MMDD>_ingest/ 标准 v2.1 数据集，之后直接走 run.py / 01→08。
适配器代码在 adapters/（umi / ego / sim / teleop），每源独立 detect/ingest。
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SOURCES = {"umi", "ego", "sim", "teleop"}
DEFAULT_TAIL = "_ingest"  # 输出名后缀，避免与源/既有命名冲突


def load_adapter(name: str):
    if name == "umi":
        from adapters import umi
        return umi
    if name == "ego":
        from adapters import ego
        return ego
    if name == "sim":
        from adapters import sim
        return sim
    if name == "teleop":
        from adapters import teleop
        return teleop
    raise SystemExit(f"未知源类型 {name}，可选 {sorted(SOURCES)}")


def run_demo(name: str, tmp_root: Path, extra: dict) -> tuple[Path, dict]:
    """生成合成原始样例并转换（验证框架与全流程，无真实数据时的离线演示）。"""
    maker = {
        "umi": "make_umi_raw.py",
        "ego": "make_ego_raw.py",
        "sim": "make_sim_raw.py",
    }.get(name)
    if not maker:
        raise SystemExit("teleop 源无合成样例——它本来就是标准 v2.1，可直接拿 make_demo_data 产物测试")
    raw = tmp_root / f"{name}_raw_demo"
    if raw.exists():
        shutil.rmtree(raw)
    subprocess.run([sys.executable, str(ROOT / "tools" / maker), "--out", str(raw)], check=True)
    out_root = tmp_root / "datasets"
    out_root.mkdir(parents=True, exist_ok=True)
    cfg = {k: v for k, v in extra.items() if v}
    return load_adapter(name).ingest(raw, out_root, cfg)


def main() -> int:
    ap = argparse.ArgumentParser(description="INGEST：任一采集源 → 标准 LeRobot v2.1")
    ap.add_argument("--source", choices=sorted(SOURCES), required=True, help="采集源类型")
    ap.add_argument("--input", help="源原始目录（--demo 时忽略）")
    ap.add_argument("--output", help="转换输出根目录（默认 <input>/../ingest_out）")
    ap.add_argument("--task", default="", help="任务名（覆盖源 config 默认）")
    ap.add_argument("--robot", default="", help="机器人/源标识（覆盖源 config 默认）")
    ap.add_argument("--fps", type=float, default=0, help="标称帧率（覆盖源 config 默认）")
    ap.add_argument("--demo", action="store_true", help="用合成样例演示（无需真实数据）")
    ap.add_argument("--demo-root", default="", help="合成样例与输出放哪（默认系统临时目录）")
    args = ap.parse_args()

    if args.demo:
        tmp_root = Path(args.demo_root) if args.demo_root else Path(ROOT) / "ingest_demo"
        tmp_root = tmp_root.resolve()
        out, report = run_demo(args.source, tmp_root, {"task": args.task, "robot": args.robot,
                                                       "fps": args.fps or 0})
        print(f"\n[INGEST-DEMO:{args.source}] 合成样例端到端完成")
        _print_report(report)
        print(f"\n下一步（走既有处理链）:")
        print(f"  python3 pipe/01_inspect.py --input {out}")
        print(f"  python3 pipe/07_verify.py --input {out}")
        return 0

    if not args.input:
        ap.error("--input 或 --demo 必须提供一个")
    src = Path(args.input).resolve()
    adapter = load_adapter(args.source)
    ok, why = adapter.detect(src)
    if not ok:
        print(f"[INGEST] {args.source} 源目录不匹配: {why}")
        return 2

    out_root = Path(args.output).resolve() if args.output else (src.parent / "ingest_out")
    out_root.mkdir(parents=True, exist_ok=True)
    cfg = {"task": args.task, "robot": args.robot, "fps": args.fps or 0}
    out, report = adapter.ingest(src, out_root, cfg)
    print(f"\n[INGEST:{args.source}] 完成 → {out}")
    _print_report(report)
    return 0


def _print_report(r: dict) -> None:
    for k, v in r.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    sys.exit(main())
