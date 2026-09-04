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
from pipe.lib import suggest as _suggest  # noqa: E402  # 仅用于 annotate 的无端点拦截

STEPS = ["inspect", "timestamps", "clean", "merge", "convert", "verify", "pack",
         "record", "annotate"]  # 已实现
PENDING: list[str] = []           # 规划中（已全部实现）

ACTION_MENU = [
    ("inspect", "盘点(01) 每集统计/视频对齐"),
    ("timestamps", "时间戳(02) 审计丢帧/回退"),
    ("clean", "清洗质检(03) 软标记坏集"),
    ("merge", "合并(05) 勾选若干批次按坏集排除并成一份"),
    ("convert", "转换(06) v2.1→v3.0 官方转换器(自动留 v2.1 备份)"),
    ("verify", "校验(07) 结构smoke + 交付sha256清单"),
    ("pack", "打包(08) tar.gz + sha256sums.txt 交付训练机"),
    ("record", "登记 处理达标后入台账"),
    ("annotate", "标注(09) VLM 逐集质量评分+建议"),
    ("aggregate", "汇总台账"),
    ("quit", "退出"),
]

SCRIPTS = {
    "inspect": "pipe/01_inspect.py",
    "timestamps": "pipe/02_timestamps.py",
    "clean": "pipe/03_clean.py",
    "merge": "pipe/05_merge.py",
    "convert": "pipe/06_convert.py",
    "verify": "pipe/07_verify.py",
    "pack": "pipe/08_pack.py",
    "record": "ledger/record.py",
    "aggregate": "ledger/aggregate.py",
    "annotate": "pipe/09_annotate.py",
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
        # 产物/备份目录不是数据集，不入列表（产品夹 _products、06 转换留的 _old 备份）
        if child.name.endswith(("_products", "_old")):
            continue
        info = dataset_io.summarize_light(child)
        out.append(info)
    return out


def scan_roots(cfg: dict, path_override: str | None) -> list[dict]:
    """可选项列表 = batches 目录下全部 + output 目录下的 v2.1（合并产物/待转换）。

    想扫别处仍用 --path 只扫那个目录。返回绝对路径条目（group: 批次/输出）。
    """
    items = []
    root = Path(path_override or cfg.get("paths", {}).get("batches") or "").expanduser()
    if root.is_dir():
        items = scan_batches(root)
        for i in items:
            i.setdefault("group", "批次")
    if not path_override:
        out_root = Path(cfg.get("paths", {}).get("output") or "").expanduser()
        if out_root.is_dir() and out_root.resolve() != root.resolve():
            outs = [i for i in scan_batches(out_root) if i["kind"] in ("v2.1", "v3.0")]
            for i in outs:
                i["group"] = "输出"
            items = items + outs
    return items


def state_path(cfg: dict, override: str | None) -> Path:
    # --state 优先：视为确切的状态文件路径
    if override:
        return Path(override).expanduser()
    # 否则状态文件放 paths.output 目录内，避免把输出目录本身占成文件
    out = cfg.get("paths", {}).get("output")
    if out:
        base = Path(out).expanduser()
        # 兼容旧版误把 output 路径写成状态文件的情况：还原为目录 + 内部状态文件
        if base.is_file():
            _migrate_state_from_file(base)
        return base / ".dataprep_state.json"
    return ROOT / ".dataprep_state.json"


def _migrate_state_from_file(base: Path) -> None:
    """旧版本曾把 paths.output 路径直接当作状态文件占用，还原成目录并迁移状态。"""
    try:
        data = base.read_bytes()
        content = data.decode("utf-8")
    except Exception:  # noqa: BLE001
        content = "{}"
    try:
        base.unlink()          # 删掉占用的文件
        if not base.is_dir():  # 建立同名目录（run.py 之前不会真的建目录）
            one = base.parent / "__mk_placeholder__"
            one.mkdir(parents=True, exist_ok=True)
            one.rename(base)
        (base / ".dataprep_state.json").write_text(content, encoding="utf-8")
        print(f"[i] 已将原 output 路径从文件还原为目录，状态迁移至 {base / '.dataprep_state.json'}")
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] 迁移 output 路径失败（{e}），状态将写入默认位置")


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
    s = st.get(key, {})
    status = "  ".join(f"{n}✓" for n in ("inspect", "timestamps", "clean", "merge", "convert",
                                          "verify", "pack", "record", "annotate") if s.get(n))
    if info["kind"] == "not_dataset":
        return f"  [{idx}] {info['name']:<34} — 不可用: {info['reason']}"
    tag = {"v2.1": "v2.1", "v3.0": "v3.0"}.get(info["kind"], info["kind"])
    info_l = ""
    if info.get("n_episodes") is not None:
        info_l = f"{info['n_episodes']}集 {info['fps']:g}fps {info['robot_type']}"
    return f"  [{idx}] {info['name']:<32} {tag:<5} {info_l:<18} {status or '未处理'}"


def show_batches(batches: list[dict], st: dict) -> None:
    print("-" * 70)
    cur = None
    for i, info in enumerate(batches, 1):
        g = info.get("group", "批次")
        if g != cur:
            cur = g
            label = "采集批次（paths.batches）" if g == "批次" else "已处理输出（paths.output）"
            print(f"◆ {label}")
        print(fmt_row(i, info, st))
    print("-" * 70)


# ---------------------------------------------------------------- 执行
def run_script(script: str, argv: list[str], out_stream=None) -> int:
    cmd = [sys.executable, script, *argv]
    header = f"==> python3 {script} {' '.join(argv)}"
    if out_stream is not None:  # Web 场景：逐行推给 SSE
        out_stream(header)
        return _run_streamed(cmd, out_stream)
    print(header, flush=True)
    r = subprocess.run(cmd, cwd=str(ROOT))
    return r.returncode


def _run_streamed(cmd: list[str], out_stream) -> int:
    p = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    if p.stdout:
        for line in p.stdout:
            out_stream(line.rstrip("\n"))
    p.wait()
    out_stream("[exit]")
    return p.returncode


def build_action_argv(action: str, allowed: list[dict], cfg: dict, opts: dict | None) -> list[str]:
    """把"动作 + 选中批次"组装成子进程 argv，供 CLI 与 Web 共用，不带任何 input()。"""
    opts = opts or {}
    if action == "merge":
        argv = ["--inputs", *[i["path"] for i in allowed]]
        name = opts.get("out_name")
        if name:
            out_root = cfg.get("paths", {}).get("output") or "."
            argv += ["--output", str(Path(out_root).expanduser() / name)]
        if (ROOT / "config.yaml").is_file():
            argv += ["--config", str(ROOT / "config.yaml")]
        return argv
    if action == "convert":
        argv = ["--input", allowed[0]["path"]]
        if opts.get("yes"):
            argv += ["--yes"]
        return argv
    if action == "record":
        argv = ["--batch", allowed[0]["path"], "--stage", opts.get("stage") or "final"]
        if opts.get("yes"):
            argv += ["--yes"]
        if (ROOT / "config.yaml").is_file():
            argv += ["--config", str(ROOT / "config.yaml")]
        return argv
    if action == "annotate":
        argv = []
        for info in allowed:
            argv += ["--input", info["path"]]
        if (ROOT / "config.yaml").is_file():
            argv += ["--config", str(ROOT / "config.yaml")]
        return argv
    if action == "verify":
        argv = []
        for info in allowed:
            argv += ["--input", info["path"]]
        return argv
    if action == "pack":
        argv = ["--input", allowed[0]["path"]]
        out_root = cfg.get("paths", {}).get("output")
        if out_root:
            argv += ["--out", str(Path(out_root).expanduser())]
        if opts.get("overwrite"):
            argv += ["--overwrite"]
        return argv
    # inspect / timestamps / clean
    argv = []
    for info in allowed:
        argv += ["--input", info["path"]]
    return argv


def execute_action(action: str, allowed: list[dict], cfg: dict, st: dict,
                   state_f: Path, opts: dict | None = None, out_stream=None) -> int:
    """非交互执行链：前置校验 → 组 argv → 跑子进程 → 成功后标记/留痕。

    CLI 与 Web 共用；Web 传 out_stream 逐行为 SSE，CLI 传 None 打印到 stdout。
    """
    opts = opts or {}
    say = out_stream or (lambda s: print(s))
    allowed, _blocked = split_allowed(action, allowed, say)  # 前置校验提示走 say(SSE)
    if not allowed:
        say("[!] 所选批次均不满足该动作的前置条件，未执行任何操作")
        return 1
    if action == "merge" and len(allowed) < 2:
        say("[!] 合并至少需要 2 个可合并的 v2.1 批次")
        return 1
    if action == "convert" and len(allowed) != 1:
        say("[!] 转换一次只处理一个 v2.1 数据集")
        return 1
    if action == "record" and len(allowed) != 1:
        say("[!] 登记一次只处理一个批次，请重新选择")
        return 1
    if action == "pack" and len(allowed) != 1:
        say("[!] 打包一次只处理一个数据集")
        return 1
    if action == "annotate":
        _ac = _suggest.annotate_config(cfg)
        if not _ac["ok"]:
            say(f"[!] 标注被拦截：{_ac['reason']}")
            return 1
    argv = build_action_argv(action, allowed, cfg, opts)
    rc = run_script(SCRIPTS[action], argv, out_stream=say)
    if rc == 0:
        try:
            for info in allowed:
                mark(st, info["path"], action)
                note = opts.get("stage") or "" if action == "record" else ""
                record_op(state_f, info, action, note=note)
            save_state(state_f, st)
        except OSError as e:
            say(f"[WARN] 动作成功但状态/留痕写入失败（{e}），不影响数据本身")
    return rc


def pick_multi(batches: list[dict], prompt: str) -> list[dict]:
    while True:
        try:
            raw = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            return []
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


# ---------------------------------------------------------------- 前置校验
def action_kind_hint(action: str, info: dict) -> str | None:
    """返回该批次执行该动作被挡住的原因；None=允许。自由编排：任意批次都可选中，
    但版本不满足的动作给出明确提示，不静默跳过、也不用 0 集误导。"""
    kind = info.get("kind")
    if action in ("inspect", "timestamps", "clean", "record", "annotate", "verify", "pack"):
        return None if kind in ("v2.1", "v3.0") else "该步骤需要 v2.1/v3.0 数据集（exclude 软标记，不删源）"
    if action == "convert":
        if kind == "v3.0":
            return "已是 v3.0，无需转换（可直接校验/登记）"
        return None if kind == "v2.1" else "转换只针对 v2.1 数据集"
    if action == "merge":
        return None if kind == "v2.1" else "合并源需 v2.1"
    return None


def ops_file(state_f: Path) -> Path:
    return state_f.parent / ".dataprep_ops.jsonl"


def record_op(state_f: Path, info: dict, action: str, note: str = "") -> None:
    """操作留痕：每次动作追加一行到 .dataprep_ops.jsonl，可在 UI/命令行回看每批处理链。"""
    try:
        p = ops_file(state_f)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "time": datetime.now().isoformat(timespec="seconds"),
                "action": action,
                "name": info.get("name"),
                "path": info.get("path"),
                "kind": info.get("kind"),
                "note": note,
            }, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001
        pass


def split_allowed(action: str, selected: list[dict], say=None) -> tuple[list[dict], list[tuple[dict, str]]]:
    """把选中批次分成 (可执行, 被挡<批次,原因>)。被挡的会打印提示但不会执行。
    say=callable 时用它输出（Web 走 SSE），否则 print 到 stdout。"""
    say = say or (lambda s: print(s))
    allowed: list[dict] = []
    blocked: list[tuple[dict, str]] = []
    for info in selected:
        hint = action_kind_hint(action, info)
        if hint:
            say(f"[i] {info['name']} 跳过: {hint}")
            blocked.append((info, hint))
        else:
            allowed.append(info)
    return allowed, blocked


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

    # CLI：用 input() 收集本动作的可选参数，其余走共用的 execute_action
    opts: dict = {}
    if action == "merge":
        if args.output:
            opts["out_name"] = args.output
        elif sys.stdin.isatty():
            out_root = cfg.get("paths", {}).get("output")
            if out_root:
                try:
                    name = input("输出目录名（回车自动生成）> ").strip()
                except (EOFError, KeyboardInterrupt):
                    name = ""
                if name:
                    opts["out_name"] = name
    elif action == "convert":
        if not sys.stdin.isatty():  # 非交互场景自动确认
            opts["yes"] = True
    elif action == "record":
        stage = args.stage
        if not stage and sys.stdin.isatty():
            try:
                ans = input("登记阶段?  1) final(处理后,默认)  2) raw(原始批次)  [回车=final] ").strip()
            except (EOFError, KeyboardInterrupt):
                ans = ""
            stage = "raw" if ans == "2" else "final"
        opts["stage"] = stage or "final"
    if action == "pack":
        if args.overwrite:
            opts["overwrite"] = True
    return execute_action(action, selected, cfg, st, state_f, opts)


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
        selected = [dataset_io.summarize_light(Path(d)) for d in args.dirs]
    else:
        items = scan_roots(cfg, args.path)
        if not items:
            print("[ERROR] 未扫到数据。检查 config.yaml 的 paths.batches，或用 --path / --dirs 指定")
            return 1
        if not args.indices:
            print(f"[ERROR] 请给编号，例如: python3 run.py {args.cmd} 1,2（编号见 run.py list）")
            return 1
        selected = resolve_indices(items, args.indices)
        if selected is None:
            return 1
    return do_action(args.cmd, selected, cfg, args, st, state_f)


def cmd_list(cfg: dict, args: argparse.Namespace, st: dict) -> int:
    items = scan_roots(cfg, args.path)
    if not items:
        print("[ERROR] 没有可列出的数据：先配置 config.yaml 的 paths.batches（合并产物放 paths.output），"
              "或用 --path 指定目录")
        return 1
    show_batches(items, st)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="run.py", description="innov-dataprep 总入口")
    ap.add_argument("cmd", nargs="?", default="menu",
                    help="list / inspect / timestamps / clean / merge / convert / verify / pack / record / annotate / aggregate；空=交互菜单")
    ap.add_argument("indices", nargs="*", help="数据编号（list 里看到的），如 1,3")
    ap.add_argument("--path", default=None, help="扫描哪个目录（默认 config paths.batches）")
    ap.add_argument("--dirs", nargs="+", default=None, help="直接给数据集路径，跳过扫描")
    ap.add_argument("--stage", choices=["final", "raw"], default=None, help="record 阶段")
    ap.add_argument("--output", default=None, help="合并输出目录（merge 用，默认自动命名）")
    ap.add_argument("--overwrite", action="store_true", help="覆盖已存在的输出（merge/pack 用）")
    ap.add_argument("--yes", action="store_true", help="非交互场景自动确认（convert/record 用）")
    ap.add_argument("--aggregate-dir", default=None, help="汇总台账所在目录")
    ap.add_argument("--state", default=None, help="状态文件路径")
    args = ap.parse_args()

    cfg = load_config()
    state_f = state_path(cfg, args.state)
    st = load_state(state_f)

    if args.cmd == "menu":
        items = scan_roots(cfg, args.path)
        if not items:
            print("[!] 没有可处理的数据（先配置 config.yaml 的 paths.batches，或 --path 指定目录后重试）")
            return 1
        while True:
            show_batches(items, st)
            print("你想做什么？")
            for i, (name, desc) in enumerate(ACTION_MENU, 1):
                tag = "" if name in STEPS + ["aggregate", "quit"] else " (待实现)"
                print(f"    {i}. {name:<10} {desc}{tag}")
            try:
                choice = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见。")
                return 0
            if choice in ("quit", "q", ""):
                print("再见。")
                return 0
            if not choice.isdigit() or not (1 <= int(choice) <= len(ACTION_MENU)):
                print(f"[!] 请输入菜单编号 1-{len(ACTION_MENU)}（输入 q 退出）")
                continue
            action = ACTION_MENU[int(choice) - 1][0]
            if action in PENDING:
                do_action(action, [], cfg, args, st, state_f)
                continue
            if action == "aggregate":
                do_action("aggregate", [], cfg, args, st, state_f)
                continue
            selected = pick_multi(items, f"选择要『{action}』的数据编号（多选逗号分隔，回车取消）> ")
            if not selected:
                print("[i] 已取消")
                continue
            do_action(action, selected, cfg, args, st, state_f)
        return 0

    if args.cmd == "list":
        return cmd_list(cfg, args, st)
    if args.cmd not in SCRIPTS:
        print(f"[ERROR] 未知命令: {args.cmd}（可用: list / inspect / timestamps / clean / merge / convert / verify / pack / record / annotate / aggregate）")
        return 2
    return cmd_clean(cfg, args, state_f, st)


if __name__ == "__main__":
    sys.exit(main())