#!/usr/bin/env python3
"""06 转换：v2.1 → v3.0（调用官方 lerobot convert_dataset_v21_to_v30.py，本地转换）。

行为（官方转换器语义，已核源码）：
  - 写到 <名字>_v30/ 成功后，原 v2.1 目录改名为 <名字>_old/（保留备份），
    v3.0 摆回原名路径；
  - 必须 --push-to-hub=false（本地转，不传 Hub）；
  - 硬性前提：codebase_version == "v2.1" 且 meta/episodes_stats.jsonl 存在
    （缺则本脚本自动从 parquet 补算后继续）。

调用方式自动探测（各机器 lerobot 装法不同）：
  1) python -m lerobot.scripts.convert_dataset_v21_to_v30
  2) 定位 lerobot 包内 scripts/convert_dataset_v21_to_v30.py 全路径
  3) 常见源码目录（~/robodeploy/lerobot、~/lerobot）

用法:
    python3 pipe/06_convert.py --input <v2.1目录> [--yes] [--check]
    # --check  只做预检（不转换）：v2.1? / stats? / 转换器在哪?
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipe.lib import dataset_io  # noqa: E402


# ---------------------------------------------------------------- 探测官方转换器
def _imports_lerobot(python: str) -> bool:
    r = subprocess.run([python, "-c", "import lerobot"], capture_output=True, text=True)
    return r.returncode == 0


def probe_converter(python: str) -> tuple[list[str] | None, str]:
    """返回 ([argv前缀], 描述) 或 (None, 原因)。"""
    # 1) 模块调用（lerobot 已 pip 安装时最可靠）
    if _imports_lerobot(python):
        r = subprocess.run([python, "-c", "import lerobot.scripts.convert_dataset_v21_to_v30"],
                           capture_output=True, text=True)
        if r.returncode == 0:
            return ([python, "-m", "lerobot.scripts.convert_dataset_v21_to_v30"], "模块调用")
    # 2) 包内脚本全路径
    r = subprocess.run([python, "-c",
                        "import lerobot, pathlib; print(pathlib.Path(lerobot.__file__).parent / 'scripts' / 'convert_dataset_v21_to_v30.py')"],
                       capture_output=True, text=True)
    if r.returncode == 0 and r.stdout.strip():
        p = Path(r.stdout.strip())
        if p.is_file():
            return ([python, str(p)], f"包内脚本 {p}")
    # 3) 常见源码目录（如 ~/lerobot、~/robodeploy/lerobot；要求当前环境能 import lerobot）
    if not _imports_lerobot(python):
        return (None, "当前环境无法 import lerobot，无法使用官方转换器。"
                      "请确认在装有 lerobot 的 conda 环境运行（采集机: conda activate lerobot_arx_sdk311）")
    for base in (Path.home() / "robodeploy" / "lerobot", Path.home() / "lerobot"):
        for cand in (base / "src" / "lerobot" / "scripts" / "convert_dataset_v21_to_v30.py",
                     base / "lerobot" / "scripts" / "convert_dataset_v21_to_v30.py"):
            if cand.is_file():
                return ([python, str(cand)], f"源码目录 {cand}")
    return (None, "lerobot 已装但找不到转换器脚本（版本过老？），可升级 lerobot 或换源码运行")


# ---------------------------------------------------------------- 补算 stats
def backfill_stats(ds: Path) -> int:
    """缺 episodes_stats.jsonl 时从 parquet 补算（官方转换器硬性需要）。"""
    rows = []
    for p in dataset_io.discover_episodes(ds):
        ep = dataset_io.episode_index(p)
        df = pd.read_parquet(p)
        rows.append(dataset_io.compute_episode_stats(df, ep))
    rows.sort(key=lambda r: r["episode_index"])
    out = ds / "meta" / "episodes_stats.jsonl"
    out.write_text("\n".join(__import__("json").dumps(r, ensure_ascii=False) for r in rows) + "\n",
                   encoding="utf-8")
    return len(rows)


# ---------------------------------------------------------------- 预检
def preflight(ds: Path) -> dict:
    kind, reason = dataset_io.detect_dataset(ds)
    if kind != "v2.1":
        return {"ok": False, "reason": f"不是 v2.1 数据集（{reason}）。转换器只收 v2.1。"}
    info = dataset_io.read_meta(ds)["info"]
    n_ep = len(dataset_io.discover_episodes(ds))
    stats_p = ds / "meta" / "episodes_stats.jsonl"
    if not stats_p.is_file():
        n = backfill_stats(ds)
        print(f"[i] 缺 meta/episodes_stats.jsonl，已自动从 parquet 补算 {n} 集（官方转换器需要）")
    return {"ok": True, "n_episodes": n_ep,
            "total_frames": info.get("total_frames"), "name": ds.name}


def main() -> int:
    ap = argparse.ArgumentParser(description="06 转换：v2.1 → v3.0（官方转换器，本地转换留备份）")
    ap.add_argument("--input", required=True, help="v2.1 数据集目录（转换后 v3.0 回到此路径，原 v2.1 备份为 <名字>_old）")
    ap.add_argument("--check", action="store_true", help="只预检不转换")
    ap.add_argument("--yes", action="store_true", help="跳过确认（非交互场景用）")
    ap.add_argument("--data-size-mb", type=int, default=None)
    ap.add_argument("--video-size-mb", type=int, default=None)
    args = ap.parse_args()

    ds = Path(args.input).expanduser().resolve()
    if not ds.is_dir():
        print(f"[ERROR] 目录不存在: {ds}")
        return 1

    pf = preflight(ds)
    if not pf["ok"]:
        print(f"[ERROR] {pf['reason']}")
        return 1
    print(f"[OK] 预检通过: {pf['name']}（v2.1, {pf['n_episodes']} 集, {pf['total_frames']} 帧）")

    if args.check:
        print("[i] --check 模式，未执行转换。")
        return 0

    # 探测官方转换器
    python = sys.executable
    argv, desc = probe_converter(python)
    if argv is None:
        print(f"[ERROR] 官方转换器 {desc}。")
        print("        请在装有 lerobot 的采集机环境跑，并先确认: python -c 'import lerobot' 可用；")
        print("        本机 probe: python -m lerobot.scripts.convert_dataset_v21_to_v30")
        return 1
    print(f"[i] 官方转换器: {desc}")

    old_root = ds.parent / f"{ds.name}_old"
    if old_root.is_dir():
        print(f"[i] 发现历史备份 {old_root.name}（上次转换残留），官方转换器会先还原再转")

    if not args.yes:
        try:
            ans = input(f"将把 {ds.name} 转换为 v3.0（原 v2.1 保留为 {ds.name}_old）。继续? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            ans = ""
        if ans not in ("y", "yes", "是"):
            print("已取消。")
            return 0

    cmd = [*argv, "--repo-id", f"local/{ds.name}",
           "--root", str(ds), "--push-to-hub", "false"]
    if args.data_size_mb:
        cmd += ["--data-file-size-in-mb", str(args.data_size_mb)]
    if args.video_size_mb:
        cmd += ["--video-file-size-in-mb", str(args.video_size_mb)]
    print(f"\n==> {' '.join(cmd[:2] + ['…', *cmd[3:]])}", flush=True)

    import logging
    logging.disable(logging.DEBUG)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"\n[ERROR] 官方转换器返回 {r.returncode}，请查看上方日志。")
        print("       常见原因: 网络/依赖缺失(datasets/hf_hub/jsonlines)；数据问题请看报错点。")
        return r.returncode

    print(f"\n[OK] 转换完成 -> {ds} 现在是 v3.0")
    print(f"[OK] 原 v2.1 备份: {old_root}")
    print("[i] 下一步: 07 校验 / 登记(final) / 08 打包。v2.1 数据不再被 03/05 处理（已 v3.0）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())