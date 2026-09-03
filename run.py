#!/usr/bin/env python3
"""总入口：一条命令搞定所有步骤（菜单点选 / 短命令），避免记长命令、防选错。

原则：
  - 扫描范围 = config.paths.batches 目录下的【一层子目录】（不会全机器扫）；
    想扫别处用 --path <目录>，想完全绕开扫描用 --dirs <路径...>；
  - 每个子目录按"是不是 v2.1 数据集"识别，识别不了的列出原因、不可选；
  - 处理只对你勾选的批次生效（想处理哪批就哪批）；合并(05)可勾任意组合（待实现）；
  - 每批处理状态记在 manifest（--state 指定，默认 config.paths.output/.dataprep_state.json）。

用法:
    python3 run.py                      # 交互菜单（推荐）
    python3 run.py list                 # 列出批次+状态
    python3 run.py clean 1,2            # 清洗批次 1、2（编号见 list）
    python3 run.py record 3             # 登记批次 3（处理后）
    python3 run.py inspect 1 --path /x # 扫别的目录
    python3 run.py clean --dirs /a /b  # 直接给路径，不扫描
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from pipe.lib import dataset_io  # noqa: E402

STEPS = ["inspect", "timestamps", "clean", "record"]          # 已实现
PENDING = ["merge", "convert", "verify", "pack"]               # 规划中（05-08）

ACTION_MENU = [
    ("inspect", "盘点(01) 每集统计/视频对齐"),
    ("timestamps", "时间戳(02) 审计丢帧/回退"),
    ("clean", "清洗质检(03) 软标记坏集"),
    ("merge", "合并(05) 按处置清单排除坏集 [待实现]"),
    ("convert", "转换 v2.1→v3.0 [待实现]"),
    ("record", "登记 处理达标后入台账"),
    ("aggregate", "汇总台账"),
    ("quit", "退出"),
]

SCRIPTS = {
    "inspect": "pipe/01_inspect.py",
    "timestamps": "pipe/02_timestamps.py",
    "clean": "pipe/03_clean.py",
    "record": "ledger/record.py",
    "aggregate": "ledger/aggregate.py",
}


# ---------------------------------------------------------------- 配置与状态
def load_config() -> dict:
    p = ROOT / "config.yaml"
    if not p.is_file():
        return {}
    try:
        import yaml
        with open(p, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:  # noqa: BLE001
        print(f"[WARN] config.yaml 解析失败，按无配置处理: {p}")
        return {}


def scan_batches(root: Path) -> list[dict]:
    """扫描 root 下的一层子目录，逐个识别并轻量摘要（统一绝对路径）。"""
    out = []
    root = root.expanduser().resolve()
    if not root.is_dir():
        print(f"[ERROR] 批次目录不存在: {root}")
        return out
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        info = dataset_io.summarize_light(child)
        out.append(info)
    return out


def state_path(cfg: dict, override: str | None) -> Path:
    p = override or cfg.get("paths", {}).get("output")
    return Path(p).expanduser() if p else (ROOT / ".dataprep_state.json")


def load_state(p: Path) -> dict:
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
    return {}


def save_state(p: Path, st: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")


def mark(st: dict, path: str, step: str) -> None:
    st.setdefault(path, {})[step] = datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------- 展示
def fmt_row(idx: int, info: dict, st: dict) -> str:
    key = info["path"]
    if info["kind"] != "v2.1":
        return f"  [{idx}] {info['name']:<34} — 不可用: {info['reason']}"
    s = st.get(key, {})
    status = "  ".join(f"{n}✓" for n in ("inspect", "timestamps", "clean", "record") if s.get(n))
    info_l = ""
    if info.get("n_episodes") is not None:
        info_l = f"{info['n_episodes']}集 {info['fps']:g}fps {info['robot_type']}"
    return f"  [{idx}] {info['name']:<34} {info_l:<18} {status or '未处理'}"


def show_batches(batches: list[dict], st: dict) -> None:
    print("-" * 70)
    print("批次目录下的数据（只扫 path.batches 这一层）：")
    for i, info in enumerate(batches, 1):
        print(fmt_row(i, info, st))
    print("-" * 70)


# ---------------------------------------------------------------- 执行
def run_script(script: str, argv: list[str]) -> int:
    print(f"\n==> python3 {script} {' '.join(argv)}", flush=True)
    r = subprocess.run([sys.executable, script, *argv], cwd=str(ROOT))
    return r.returncode


def pick_multi(batches: list[dict], prompt: str) -> list[dict]:
    while True:
        raw = input(prompt).strip()
        if not raw:
            return []
        try:
            idxs = [int(x) for x in raw.replace(" ", ",").split(",") if x]
        except ValueError:
            print("[!] 请输入编号，多选用逗号分隔（如 1,3）")
            continue
        bad = [i for i in idxs if not (1 <= i <= len(batches))]
        if bad:
            print(f"[!] 编号越界: {bad}")
            continue
        return [batches[i - 1] for i in dict.fromkeys(idxs)]


def do_action(action: str, selected: list[dict], cfg: dict, args: argparse.Namespace,
              st: dict, state_f: Path) -> int:
    if action in PENDING:
        print(f"[i] 『{action}』尚未实现（规划中），敬请期待")
        return 0
    if action == "aggregate":
        ledger = Path(cfg.get("paths", {}).get("ledger") or "data_catalog.csv")
        d = args.aggregate_dir or str(ledger.parent)
        print(f"汇总目录: {d}（找 data_catalog*.csv）")
        return run_script(SCRIPTS["aggregate"], ["--dir", d])
    if action == "record":
        if len(selected) != 1:
            print("[!] 登记一次只处理一个批次，请重新选择")
            return 1
        info = selected[0]
        if info["kind"] != "v2.1":
            print(f"[!] {info['name']} 不是 v2.1 数据集，无法登记")
            return 1
        stage = args.stage
        if not stage:
            stage = input("登记阶段?  1) final(处理后,默认)  2) raw(原始批次)  [回车=final] ").strip()
            stage = "raw" if stage == "2" else "final"
        argv = ["--batch", info["path"], "--stage", stage]
        if (ROOT / "config.yaml").is_file():
            argv += ["--config", str(ROOT / "config.yaml")]
        rc = run_script(SCRIPTS["record"], argv)
        if rc == 0:
            mark(st, info["path"], "record")
            save_state(state_f, st)
        return rc
    # inspect / timestamps / clean
    argv = []
    for info in selected:
        if info["kind"] != "v2.1":
            print(f"[!] 跳过 {info['name']}: {info['reason']}")
            continue
        argv += ["--input", info["path"]]
    if not argv:
        print("[!] 没有可处理的 v2.1 批次")
        return 1
    rc = run_script(SCRIPTS[action], argv)
    if rc == 0:
        for info in selected:
            if info["kind"] == "v2.1":
                mark(st, info["path"], action)
        save_state(state_f, st)
    return rc


# ---------------------------------------------------------------- 命令入口
def resolve_indices(batches: list[dict], tokens: list[str]) -> list[dict] | None:
    idxs: list[int] = []
    for tok in tokens:
        for piece in tok.replace(" ", ",").split(","):
            piece = piece.strip()
            if not piece:
                continue
            if not piece.isdigit():
                print(f"[!] 无效编号: {piece!r}")
                return None
            idxs.append(int(piece))
    if any(not (1 <= i <= len(batches)) for i in idxs):
        print(f"[!] 编号越界（可选 1-{len(batches)}）")
        return None
    return [batches[i - 1] for i in dict.fromkeys(idxs)]


def cmd_clean(cfg: dict, args: argparse.Namespace, state_f: Path, st: dict) -> int:
    if args.dirs:
        batches = [dataset_io.summarize_light(Path(d)) for d in args.dirs]
        selected = batches
    else:
        root = Path(args.path or cfg.get("paths", {}).get("batches") or "")
        batches = scan_batches(root) if root else []
        if not batches:
            print("[ERROR] 未扫到批次。检查 config.yaml 的 paths.batches，或用 --path / --dirs 指定")
            return 1
        if not args.indices:
            print(f"[ERROR] 请给批次编号，例如: python3 run.py {args.cmd} 1,2")
            print("       先跑 python3 run.py list 查看编号")
            return 1
        selected = resolve_indices(batches, args.indices)
        if selected is None:
            return 1
    return do_action(args.cmd, selected, cfg, args, st, state_f)


def cmd_list(cfg: dict, args: argparse.Namespace, st: dict) -> int:
    root = Path(args.path or cfg.get("paths", {}).get("batches") or "").expanduser()
    if not root or not root.is_dir():
        print(f"[ERROR] 批次目录不可用: {root}（先配置 config.yaml paths.batches，或 --path 指定）")
        return 1
    show_batches(scan_batches(root), st)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="run.py", description="innov-dataprep 总入口")
    ap.add_argument("cmd", nargs="?", default="menu",
                    help="list / inspect / timestamps / clean / record / aggregate；空=交互菜单")
    ap.add_argument("indices", nargs="*", help="批次编号（list 里看到的），如 1,3")
    ap.add_argument("--path", default=None, help="扫描哪个目录（默认 config paths.batches）")
    ap.add_argument("--dirs", nargs="+", default=None, help="直接给数据集路径，跳过扫描")
    ap.add_argument("--stage", choices=["final", "raw"], default=None, help="record 阶段")
    ap.add_argument("--aggregate-dir", default=None, help="汇总台账所在目录")
    ap.add_argument("--state", default=None, help="状态文件路径")
    args = ap.parse_args()

    cfg = load_config()
    state_f = state_path(cfg, args.state)
    st = load_state(state_f)

    if args.cmd == "menu":
        root = Path(args.path or cfg.get("paths", {}).get("batches") or "")
        batches = scan_batches(root) if root else []
        if not batches:
            print("[!] 未扫到批次（先配置 config.yaml paths.batches，或 --path 指定目录后重试）")
            return 1
        while True:
            show_batches(batches, st)
            print("你想做什么？")
            for i, (name, desc) in enumerate(ACTION_MENU, 1):
                tag = "" if name in STEPS + ["aggregate", "quit"] else " (待实现)"
                print(f"    {i}. {name:<10} {desc}{tag}")
            try:
                choice = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见。")
                return 0
            if choice in ("8", "quit", "q", ""):
                print("再见。")
                return 0
            if not choice.isdigit() or not (1 <= int(choice) <= len(ACTION_MENU)):
                print("[!] 请输入菜单编号 1-8")
                continue
            action = ACTION_MENU[int(choice) - 1][0]
            if action in PENDING:
                do_action(action, [], cfg, args, st, state_f)
                continue
            if action == "aggregate":
                do_action("aggregate", [], cfg, args, st, state_f)
                continue
            selected = pick_multi(batches, f"选择要『{action}』的批次编号（多选逗号分隔，回车取消）> ")
            if not selected:
                print("[i] 已取消")
                continue
            do_action(action, selected, cfg, args, st, state_f)
        return 0

    if args.cmd == "list":
        return cmd_list(cfg, args, st)
    if args.cmd not in SCRIPTS:
        print(f"[ERROR] 未知命令: {args.cmd}（可用: list / inspect / timestamps / clean / record / aggregate）")
        return 2
    return cmd_clean(cfg, args, state_f, st)


if __name__ == "__main__":
    sys.exit(main())