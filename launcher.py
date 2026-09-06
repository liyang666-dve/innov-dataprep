#!/usr/bin/env python3
"""数据处理台 桌面启动器 —— 零安装"软件"形态（仅标准库）。

双击 start_desk.bat → 本脚本后台运行：
  1) 起本地 HTTP 服务（127.0.0.1:PORT）：
     - 静态托管工作台（数据系统目录，自动打开 embodied-data-workspace/index.html）
     - /api/hello            引擎在线探测
     - /api/batches          扫描批次/源（config paths.batches + ingest_demo）
     - /api/run              {tool, target} 执行 INGEST / 01 / 03 / 07 / 08 / record
  2) 自动用 Edge --app 打开独立应用窗口（无地址栏，观感=桌面软件）；
     Edge 不存在时回退默认浏览器。

依赖：仅 Python 标准库。引擎脚本（00_ingest/01..08/ledger）由 ENGINE_PY 指定的
Python 执行（默认探测 D:/miniconda/envs/lerobot/python.exe，Linux 回退 python3）。

用法:
    pythonw launcher.py                 # 桌面模式（自动开 Edge 窗口）
    python  launcher.py --no-open       # 只起服务不弹窗（调试）
    python  launcher.py --port 8123 --engine-py /path/to/python
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATADIR = ROOT.parent                     # 数据系统目录（两个仓库并列）
WORKSPACE = DATADIR / "embodied-workspace"
DEFAULT_BATCHES = ROOT / "ingest_demo" / "datasets"
LOG_MAX = 6000                            # /api/run 返回日志上限(字符)
PORT = 8017

ENGINE_CANDIDATES = [
    os.environ.get("INNOV_PYTHON", ""),
    "D:/miniconda/envs/lerobot/python.exe",
    "D:/miniconda/envs/lerobot_arx_sdk311/python.exe",
    sys.executable,
]


def find_engine_py() -> str:
    for c in ENGINE_CANDIDATES:
        if c and Path(c).is_file():
            return c
    return "python3"


def load_config() -> dict:
    """读 innov-dataprep/config.yaml；yaml 缺失时退化为逐行解析 paths。"""
    cfg = {}
    path = ROOT / "config.yaml"
    if not path.is_file():
        return cfg
    try:
        import yaml  # noqa: PLC0415
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if cfg.get("paths"):
            return cfg
    except Exception:
        pass
    # 轻量回退：只取 paths.batches / paths.ledger 两行
    txt = path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"^\s*batches:\s*['\"]?([^#'\"\s]+)", txt, re.M)
    if m:
        cfg.setdefault("paths", {})["batches"] = m.group(1)
    m = re.search(r"^\s*ledger:\s*['\"]?([^#'\"\s]+)", txt, re.M)
    if m:
        cfg.setdefault("paths", {})["ledger"] = m.group(1)
    return cfg


def batches_roots(cfg: dict) -> list[Path]:
    roots = []
    b = (cfg.get("paths") or {}).get("batches")
    if b:
        roots.append(Path(b))
    if DEFAULT_BATCHES.is_dir():
        roots.append(DEFAULT_BATCHES)
    if not roots:
        roots.append(ROOT)
    uniq: list[Path] = []
    seen: set[str] = set()
    for p in roots:
        key = str(Path(p).resolve())
        if key not in seen:
            seen.add(key)
            uniq.append(Path(p))
    return uniq


def ledger_path(cfg: dict) -> Path:
    l = (cfg.get("paths") or {}).get("ledger")
    return Path(l) if l else ROOT / "data_catalog.csv"


# ---------------------------------------------------------------- 扫描
SCAN_EXTRA = ROOT / ".scan_extra.txt"   # 用户手动添加的扫描目录（每行一个，gitignore）


def load_extra_dirs() -> list[str]:
    """读取用户手动添加的扫描目录。"""
    if not SCAN_EXTRA.is_file():
        return []
    out = []
    for line in SCAN_EXTRA.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def save_extra_dirs(dirs: list[str]) -> None:
    SCAN_EXTRA.write_text("\n".join(dirs) + ("\n" if dirs else ""), encoding="utf-8")


def _read_info(ds: Path) -> dict:
    try:
        return json.loads((ds / "meta" / "info.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


def _same_dir(a: str, b: str) -> bool:
    """路径归一后比较（用于移除扫描目录时宽容匹配）。"""
    try:
        return Path(a or "").expanduser().resolve() == Path(b or "").expanduser().resolve()
    except Exception:
        return (a or "").strip().strip("\"'") == (b or "").strip().strip("\"'")


def _stage_file(ds: Path, stage: str, name: str) -> bool:
    """轻量版 dataset_io.stage_file：_products/{stage} 优先，回退旧平铺 _stage。"""
    if (ds.parent / f"{ds.name}_products" / stage / name).is_file():
        return True
    return (ds.parent / f"{ds.name}_{stage}" / name).is_file()


def _has_pack(ds: Path) -> bool:
    d = ds.parent / f"{ds.name}_products" / "delivery"
    if not d.is_dir():
        return False
    return any(p.name.endswith("_delivery.tar.gz") for p in d.glob("*"))


def _updated_ts(ds: Path) -> float:
    """数据集/产物目录最近改动时间戳（资产树显示"x 分钟前"）。"""
    ts = ds.stat().st_mtime
    prod = ds.parent / f"{ds.name}_products"
    try:
        if prod.is_dir():
            ts = max(ts, prod.stat().st_mtime)
    except OSError:
        pass
    return round(ts, 1)


def _quality_summary(ds: Path) -> dict:
    """从 _products/{clean,verify} 读质量摘要：清洗排除数 / 校验警告数（None=未跑）。"""
    q: dict = {"excluded": None, "warnings": None, "verify_ok": None}
    clean_csv = (ds.parent / f"{ds.name}_products" / "clean" / "episode_disposition.csv")
    if not clean_csv.is_file():
        clean_csv = ds.parent / f"{ds.name}_clean" / "episode_disposition.csv"
    if clean_csv.is_file():
        try:
            lines = clean_csv.read_text(encoding="utf-8", errors="replace").splitlines()[1:]
            q["excluded"] = sum(1 for ln in lines if "exclude" in ln.lower())
        except Exception:
            pass
    ver_json = ds.parent / f"{ds.name}_products" / "verify" / "verify_report.json"
    if not ver_json.is_file():
        ver_json = ds.parent / f"{ds.name}_verify" / "verify_report.json"
    if ver_json.is_file():
        try:
            v = json.loads(ver_json.read_text(encoding="utf-8"))
            q["warnings"] = v.get("n_warnings")
            if q["warnings"] is None and isinstance(v.get("warnings"), list):
                q["warnings"] = len(v["warnings"])
            q["verify_ok"] = bool(v.get("ok", v.get("passed", None)))
        except Exception:
            pass
    return q


def scan_candidates(cfg: dict) -> list[dict]:
    """扫描产出候选：数据集（v2.1/v3.0）与 原始源（带 config.json + source_type）。"""
    out: list[dict] = []
    seen: set[str] = set()

    def add(d: Path) -> None:
        key = str(d.resolve())
        if key in seen:
            return
        seen.add(key)
        info = _read_info(d)
        sm = info.get("source_meta") or {}
        source_type = sm.get("source_type") or ""
        is_ds = bool(info.get("codebase_version")) or bool((d / "meta").is_dir())
        if not is_ds and (d / "config.json").is_file():
            try:
                c = json.loads((d / "config.json").read_text(encoding="utf-8"))
            except Exception:
                c = {}
            st = c.get("source_type") or source_type
            if st:
                out.append({"kind": "raw", "name": d.name, "path": str(d),
                            "source_type": st, "task": c.get("task", ""),
                            "n_demos": sum(1 for x in d.iterdir() if x.is_dir()),
                            "updated_ts": _updated_ts(d),
                            "steps": {}})
            return
        if not (d / "data").is_dir():
            return
        data_dir = d / "data"
        is_v3 = bool(next(data_dir.glob("chunk-*/file-*.parquet"), None)) \
            or (d / "meta" / "episodes").is_dir()
        has_v21 = bool(next(data_dir.glob("chunk-*/episode_*.parquet"), None))
        if not (is_v3 or has_v21):
            return
        kind = "v3.0" if is_v3 else "v2.1"
        frames = 0
        try:
            for line in (d / "meta" / "episodes.jsonl").read_text(encoding="utf-8").splitlines():
                if line.strip():
                    frames += int(json.loads(line).get("length", 0))
        except Exception:
            pass
        cam_feats = list((info.get("features") or {}).keys())
        out.append({
            "kind": "dataset", "name": d.name, "path": str(d),
            "format": kind, "source_type": source_type or (sm.get("adapter", "").split(".")[0] or "teleop"),
            "robot": info.get("robot_type", "?"), "fps": info.get("fps"),
            "episodes": info.get("total_episodes"), "frames": frames,
            "cams": cam_feats, "task": d.name.split("_")[0] if not source_type else d.name,
            "updated_ts": _updated_ts(d),
            "quality": _quality_summary(d),
            "steps": {
                "ingest": bool(source_type),
                "clean": _stage_file(d, "clean", "episode_disposition.csv"),
                "verify": _stage_file(d, "verify", "verify_report.json"),
                "pack": _has_pack(d),
            },
        })

    roots = batches_roots(cfg) + [Path(x) for x in load_extra_dirs()]
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        # 兼容：root 本身是一个数据集（--input 单目录）
        if (root / "meta").is_dir():
            add(root)
            continue
        for child in sorted(root.iterdir()):
            if not child.is_dir() or child.name.endswith("_products") \
               or child.name in ("verify_work", "dataprep_out", "_old"):
                continue
            # ingest_demo/datasets 这类"套一层"容器目录：往下扫一层
            if (child / "meta").is_dir() or (child / "config.json").is_file():
                add(child)
            elif child.name in ("datasets",):
                for c2 in sorted(child.iterdir()):
                    if c2.is_dir():
                        add(c2)
    # demo 原始源（ingest_demo/*_raw_demo 等带 config.json 的非数据集目录）
    demo = ROOT / "ingest_demo"
    if demo.is_dir():
        for child in sorted(demo.iterdir()):
            if child.is_dir() and (child / "config.json").is_file() \
               and not (child / "meta").is_dir():
                add(child)
    return out


# ---------------------------------------------------------------- 引擎执行
def run_tool(tool: str, target: str, cfg: dict, engine_py: str) -> dict:
    tgt = Path(target)
    parent = tgt.parent
    prods = tgt.parent / f"{tgt.name}_products"
    name = tgt.name

    scripts = {
        "ingest": [ROOT / "pipe" / "00_ingest.py",
                   "--source", "", "--input", str(tgt), "--output", str(parent)],
        "inspect": [ROOT / "pipe" / "01_inspect.py", "--input", str(tgt)],
        "clean": [ROOT / "pipe" / "03_clean.py", "--input", str(tgt),
                  "--out", str(prods / "clean")],
        "verify": [ROOT / "pipe" / "07_verify.py", "--input", str(tgt)],
        "pack": [ROOT / "pipe" / "08_pack.py", "--input", str(tgt),
                 "--out", str(prods / "delivery")],
        "record": [ROOT / "ledger" / "record.py", "--batch", str(tgt), "--stage", "final",
                   "--yes", "--out", str(ledger_path(cfg))],
        "dehand": [ROOT / "pipe" / "10_hand_remove.py", "--input", str(tgt)],
    }
    if tool not in scripts:
        return {"ok": False, "log": f"未知工具 {tool}"}

    cmd = scripts[tool]
    if tool == "ingest":
        # 从源目录 config.json 推断 source_type
        st = ""
        try:
            c = json.loads((tgt / "config.json").read_text(encoding="utf-8"))
            st = c.get("source_type", "")
        except Exception:
            pass
        if st not in ("umi", "ego", "sim", "teleop"):
            return {"ok": False, "log": f"无法推断源类型（config.json 缺 source_type）：{st}"}
        cmd[cmd.index("--source") + 1] = st
    elif tool == "record":
        cmd += ["--operator", "launcher"]

    log = f"$ {engine_py} {' '.join(str(x) for x in cmd)}\n"
    try:
        r = subprocess.run([engine_py] + [str(x) for x in cmd],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=900, cwd=str(ROOT))
        tail = (r.stdout or "")[-LOG_MAX:] + "\n" + (r.stderr or "")[-2000:]
        log += tail
        return {"ok": r.returncode == 0, "exit": r.returncode, "log": log[-LOG_MAX * 2:]}
    except subprocess.TimeoutExpired:
        return {"ok": False, "log": log + "\n[超时] 900s 未完成，请人工检查。"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "log": log + f"\n[执行异常] {e}"}


# ---------------------------------------------------------------- HTTP
class Handler(SimpleHTTPRequestHandler):
    engine_py: str = "python3"
    cfg: dict = {}

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(DATADIR), **kw)

    def log_message(self, *a):  # 静默
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path.startswith("/api/"):
            api = self.path[5:].split("?")[0]
            if api == "hello":
                return self._json({"ok": True, "app": "embodied-data-desk-launcher",
                                   "engine_py": self.engine_py,
                                   "workspace": str(WORKSPACE)})
            if api == "batches":
                cands = scan_candidates(self.cfg)
                return self._json({"ok": True, "count": len(cands),
                                   "items": cands,
                                   "ledger": str(ledger_path(self.cfg))})
            if api == "scandirs":
                return self._json({"ok": True, "dirs": load_extra_dirs(),
                                   "base": [str(p) for p in batches_roots(self.cfg)]})
            return self._json({"ok": False, "error": f"未知 API {api}"}, 404)
        if self.path in ("/", "/index.html"):
            self.send_response(302)
            self.send_header("Location", "/embodied-workspace/index.html")
            self.end_headers()
            return
        return super().do_GET()

    def _read_body(self) -> dict:
        try:
            n = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(n).decode("utf-8")) if n else {}
        except Exception:
            return {}

    def _norm_dir(self, raw: str) -> Path | None:
        """展开 ~ / 去引号；必须是已存在目录。"""
        raw = (raw or "").strip().strip("\"'")
        if not raw:
            return None
        p = Path(raw).expanduser().resolve()
        return p if p.is_dir() else None

    def do_POST(self):  # noqa: N802
        if self.path.startswith("/api/run"):
            body = self._read_body()
            tool, target = body.get("tool", ""), body.get("target", "")
            if not tool or not target:
                return self._json({"ok": False, "error": "缺 tool/target"}, 400)
            return self._json(run_tool(tool, target, self.cfg, self.engine_py))
        if self.path.startswith("/api/scandirs"):
            body = self._read_body()
            raw = body.get("dir", "")
            if body.get("remove"):
                dirs = load_extra_dirs()
                out = [d for d in dirs if not _same_dir(d, raw)]
                save_extra_dirs(out)
                return self._json({"ok": True, "dirs": out})
            p = self._norm_dir(raw)
            if p is None:
                return self._json({"ok": False, "error": f"目录不存在：{raw}"}, 400)
            dirs = load_extra_dirs()
            key = str(p)
            if key not in dirs:
                dirs.append(key)
                save_extra_dirs(dirs)
            return self._json({"ok": True, "dirs": dirs})
        return self._json({"ok": False, "error": "仅支持 /api/run、/api/scandirs"}, 404)


def open_window(url: str) -> None:
    edge_candidates = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    for e in edge_candidates:
        if Path(e).is_file():
            subprocess.Popen([e, f"--app={url}", "--window-size=1440,920"],
                             creationflags=0x08000000)  # CREATE_NO_WINDOW
            return
    webbrowser.open(url)


class DeskServer(ThreadingHTTPServer):
    """关闭 SO_REUSEADDR：Windows 上重复 bind 同一端口会"叠罗汉"式多实例监听，
    浏览器请求随机落到旧实例 → 出现『新功能没有、报旧错误』的诡异问题。
    置 False 后端口被占时 bind 直接失败，由 main 自动换到空闲端口。"""

    allow_reuse_address = False


def _is_launcher(url: str) -> bool:
    """探测该地址是否已有本 launcher 在跑（app 名匹配），避免重复起服务。"""
    import urllib.request  # noqa: PLC0415
    try:
        with urllib.request.urlopen(url + "api/hello", timeout=0.6) as r:
            data = json.loads(r.read().decode("utf-8") or "{}")
            return data.get("app") == "embodied-data-desk-launcher"
    except Exception:  # noqa: BLE001
        return False


def find_running(port: int) -> str | None:
    """从 port 起小范围探测（默认端口与相邻几个），已有实例则返回其 URL。"""
    for p in range(port, port + 6):
        u = f"http://127.0.0.1:{p}/"
        if _is_launcher(u):
            return u
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=PORT)
    ap.add_argument("--no-open", action="store_true", help="只起服务，不自动开窗口")
    ap.add_argument("--engine-py", default=find_engine_py(), help="引擎 Python 解释器")
    args = ap.parse_args()

    Handler.engine_py = args.engine_py
    Handler.cfg = load_config()
    cfg_path = ROOT / "config.yaml"

    if not WORKSPACE.is_dir():
        print(f"[!] 未找到工作台目录: {WORKSPACE}（launcher 应放在数据系统目录下两个仓库并列）")
        return 2

    # 幂等：已有实例在跑 → 直接开它的窗口并退出，绝不重复起服务
    running = find_running(args.port)
    if running:
        print(f"[launcher] 已有实例运行于 {running}，直接打开窗口（如需重启请先关闭旧窗口/进程）")
        if not args.no_open:
            open_window(running)
        return 0

    # 端口被占（非 launcher 的其他程序）→ 自动递增找空闲端口
    httpd = None
    port = args.port
    for _ in range(50):
        try:
            httpd = DeskServer(("127.0.0.1", port), Handler)
            break
        except OSError:
            port += 1
    if httpd is None:
        print(f"[!] 端口 {args.port}-{args.port + 49} 均被占用，请先关闭残留进程")
        return 3
    if port != args.port:
        print(f"[i] 端口 {args.port} 被占用，已改用 {port}")

    url = f"http://127.0.0.1:{port}/"
    print(f"[launcher] 数据处理台服务已启动: {url}")
    print(f"[launcher] 引擎 Python: {Handler.engine_py}")
    print(f"[launcher] 工作台: {WORKSPACE}")
    if cfg_path.is_file():
        print(f"[launcher] 配置: {cfg_path}（paths.batches 决定扫描范围）")
    else:
        print("[launcher] 提示: 无 config.yaml，扫描默认 ingest_demo/datasets")
    if not args.no_open:
        open_window(url)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[launcher] 已停止")
    return 0


if __name__ == "__main__":
    sys.exit(main())
