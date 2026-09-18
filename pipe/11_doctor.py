#!/usr/bin/env python3
"""11 第二意见：调用官方生态的 lerobot-doctor 做独立体检（只读，不改数据）。

为什么需要：自研 03 覆盖的是"批次/结构/时间戳/视频帧数"，而 action 级异常
（尖峰、僵死、零方差维度、策略兼容性、URDF 动力学、per-episode 明细）由
lerobot-doctor 覆盖。两者互补 → 合并后、转换前各跑一次，互为交叉验证。

原则：
- **只读**：只跑 `check`（体检）；本脚本绝不调用 doctor 的 fix / trim（那两个会改数据）。
- **不自动杀数据**：doctor 的结论只写报告；`--merge-disposition` 时最多把
  `keep` 降级为 `review`（05 合并只排除 `exclude`，`review` 不会被删），绝不升级为 exclude。

依赖：lerobot-doctor（pip/uv 安装，不进本仓库依赖）：
    uv tool install lerobot-doctor          # 或 pip install --user lerobot-doctor
找不到时会给出明确提示；也可在 config.yaml 里指定 doctor.bin。

产物（写入 <数据集>_products/doctor/）：
    doctor.json         doctor 的原始 JSON（机器可读，含全部 checks）
    doctor_report.md    人类可读摘要（severity + 每条 message）
    disposition_note.md --merge-disposition 时说明改了哪些集

用法:
    python3 pipe/11_doctor.py --input <数据集> [--input ...]
          [--bin lerobot-doctor] [--max-episodes 20] [--merge-disposition] [--config config.yaml]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipe.lib import dataset_io, report  # noqa: E402

EP_RE = re.compile(r"[Ee]pisodes?\s+(\d+)")
BARE_EP_RE = re.compile(r"\b(\d+)\b")


# ---------------------------------------------------------------- 工具定位
def resolve_doctor(args: argparse.Namespace, cfg: dict) -> list[str] | None:
    """返回可执行命令前缀（list），找不到返回 None。"""
    cand = [str(c) for c in (args.bin, (cfg.get("doctor") or {}).get("bin")) if c]
    if not cand:
        cand = ["lerobot-doctor"]  # 默认名：先 --bin/config，再 PATH
    for c in cand:
        p = Path(str(c)).expanduser()
        if p.is_file():
            return [str(p)]
        w = shutil.which(str(c))
        if w:
            return [w]
    # 兜底：当前解释器能以 -m 方式跑（pip 装进同一环境时）
    try:
        r = subprocess.run([sys.executable, "-c", "import lerobot_doctor"],
                           capture_output=True, text=True)
        if r.returncode == 0:
            return [sys.executable, "-m", "lerobot_doctor"]
    except Exception:  # noqa: BLE001
        pass
    return None


INSTALL_HINT = """[ERROR] 未找到 lerobot-doctor（第二意见工具，本仓库不打包它）。
  任选一种装法：
    uv tool install lerobot-doctor
    python3 -m pip install --user lerobot-doctor
    pip install lerobot-doctor            # 在采集机 conda env 里
  或在 config.yaml 里指定：doctor: {bin: /path/to/lerobot-doctor}
  装不了就让这一步跳过：其它步骤不受影响。"""


# ---------------------------------------------------------------- 解析与报告
def episodes_mentioned(checks: list[dict]) -> dict[int, list[str]]:
    """从 checks 的 message 里抽出被点名的集号 → 原因列表。

    只用于"降级为 review + 写原因"，不参与 exclude。抽不到就返回空（不猜）。
    """
    out: dict[int, list[str]] = {}
    for c in checks or []:
        for m in c.get("messages") or []:
            msg = str(m.get("message") or "")
            if str(m.get("severity", "")).upper() == "PASS":
                continue
            eps: list[int] = []
            for mo in EP_RE.finditer(msg):
                eps.append(int(mo.group(1)))
            if not eps and c.get("name", "").lower().startswith("per-episode"):
                for mo in BARE_EP_RE.finditer(msg.split(":")[0]):
                    eps.append(int(mo.group(1)))
            for e in eps:
                out.setdefault(e, []).append(f"{c.get('name')}: {msg}")
    return out


def write_markdown(ds: Path, data: dict, out_root: Path) -> list[str]:
    lines = [
        f"# 第二意见（lerobot-doctor）: {ds.name}", "",
        f"- doctor 版本: {data.get('version')}",
        f"- 格式: {data.get('codebase_version')} / {data.get('format_version')}",
        f"- 集数/帧数: {data.get('total_episodes')} / {data.get('total_frames')} @ {data.get('fps')} fps",
        f"- **总体判定: {data.get('overall_severity')}**  "
        f"(PASS {data.get('summary', {}).get('PASS', 0)} / "
        f"WARN {data.get('summary', {}).get('WARN', 0)} / "
        f"FAIL {data.get('summary', {}).get('FAIL', 0)})",
        "", "> 只读体检；本步骤不修改数据、不自动排除任何集。",
        "", "## 各检查项", "",
    ]
    for c in data.get("checks") or []:
        lines.append(f"### [{c.get('severity')}] {c.get('name')}")
        for m in c.get("messages") or []:
            mark = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗"}.get(str(m.get("severity")), "-")
            lines.append(f"- {mark} {m.get('message')}")
        lines.append("")
    return lines


# ---------------------------------------------------------------- 处置清单
def disposition_path(ds: Path) -> Path | None:
    """找 03 产出的处置清单（新布局优先，兼容旧平铺布局）。"""
    cands = [
        dataset_io.products_dir(ds) / "clean" / "episode_disposition.csv",
        ds.parent / f"{ds.name}_clean" / "episode_disposition.csv",
    ]
    for p in cands:
        if p.is_file():
            return p
    return None


def merge_disposition(ds: Path, ep_reasons: dict[int, list[str]]) -> Path | None:
    """keep → review（附 doctor 原因）；exclude 不动；绝不新增 exclude。"""
    import csv
    p = disposition_path(ds)
    if p is None or not ep_reasons:
        return None
    with open(p, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
        fields = list(rows[0].keys()) if rows else []
    if "doctor" not in fields:
        fields.append("doctor")
    n = 0
    for r in rows:
        try:
            ep = int(r.get("episode", -1))
        except (TypeError, ValueError):
            continue
        why = ep_reasons.get(ep)
        if not why:
            continue
        r.setdefault("doctor", "")
        r["doctor"] = " | ".join(why)[:300]
        if str(r.get("verdict", "")).strip() == "keep":
            r["verdict"] = "review"
            n += 1
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    return p if n else None


# ---------------------------------------------------------------- 单数据集
def run_one(ds: Path, cmd: list[str], args: argparse.Namespace, cfg: dict) -> int:
    kind, reason = dataset_io.detect_dataset(ds)
    if kind not in ("v2.1", "v3.0"):
        print(f"[跳过] {ds.name} 不是 v2.1/v3.0 数据集: {reason}")
        return 1

    argv = [*cmd, "check", str(ds), "--json"]
    max_ep = args.max_episodes if args.max_episodes is not None else (cfg.get("doctor") or {}).get("max_episodes")
    if max_ep:
        argv += ["--max-episodes", str(int(max_ep))]
    print(f"[i] 运行: {' '.join(argv)}")
    r = subprocess.run(argv, capture_output=True, text=True)
    # doctor：0 = 全 PASS/WARN，1 = 有 FAIL（是结论，不是工具故障）
    if r.returncode not in (0, 1) or not (r.stdout or "").strip():
        print(f"[ERROR] lerobot-doctor 执行失败 (rc={r.returncode})")
        print((r.stderr or "")[-800:])
        return 1
    try:
        data: dict[str, Any] = json.loads(r.stdout)
    except json.JSONDecodeError:
        out_root = args.out or dataset_io.new_stage_dir(ds, "doctor")
        out_root.mkdir(parents=True, exist_ok=True)
        raw = out_root / "doctor_raw.txt"
        raw.write_text(r.stdout, encoding="utf-8")
        print(f"[WARN] doctor 输出不是 JSON，原样存到 {raw}")
        return 1

    out_root = Path(args.out) if args.out else dataset_io.new_stage_dir(ds, "doctor")
    out_root.mkdir(parents=True, exist_ok=True)
    report.write_json(out_root / "doctor.json", data)
    lines = write_markdown(ds, data, out_root)
    ep_reasons = episodes_mentioned(data.get("checks") or [])

    changed = None
    if args.merge_disposition:
        changed = merge_disposition(ds, ep_reasons)
        lines += ["", "## 处置清单联动", "",
                  (f"- 已更新处置清单 {changed}：被点名的 keep 集降级为 review"
                   if changed else "- 处置清单无需变更（未找到 03 的 episode_disposition.csv，或没有集被点名）"),
                  f"- doctor 点名的集: {sorted(ep_reasons) if ep_reasons else '无'}",
                  "- 只降级 keep→review，exclude 不动；review 不会被 05 合并删除。", ""]
    report.write_md(out_root / "doctor_report.md", lines)

    sev = data.get("overall_severity")
    print(f"[{'OK' if sev != 'FAIL' else 'WARN'}] {ds.name}: doctor 判定 {sev} "
          f"(PASS {data.get('summary', {}).get('PASS', 0)} / WARN {data.get('summary', {}).get('WARN', 0)}"
          f" / FAIL {data.get('summary', {}).get('FAIL', 0)}) → {out_root / 'doctor_report.md'}")
    for c in data.get("checks") or []:
        if str(c.get("severity")) in ("WARN", "FAIL"):
            msgs = [m.get("message") for m in (c.get("messages") or [])
                    if str(m.get("severity")) in ("WARN", "FAIL")][:2]
            for m in msgs:
                print(f"       [{c.get('severity')}] {c.get('name')}: {m}")
    if changed:
        print(f"       处置清单已更新: {changed}")
    if args.fail_on_warn and sev in ("WARN", "FAIL"):
        return 1
    if args.fail_on_fail and sev == "FAIL":
        return 1
    return 0


# ---------------------------------------------------------------- 入口
def main() -> int:
    ap = argparse.ArgumentParser(description="11 第二意见：lerobot-doctor 只读体检（不改数据）")
    ap.add_argument("--input", action="append", required=True)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--bin", default=None, help="lerobot-doctor 可执行文件路径（默认从 PATH 找）")
    ap.add_argument("--max-episodes", type=int, default=None, help="只检查前 N 集（大集强烈建议）")
    ap.add_argument("--merge-disposition", action="store_true",
                    help="把被 doctor 点名的 keep 集降级为 review（exclude 不动，绝不新增 exclude）")
    ap.add_argument("--out", default=None, help="产物目录（默认 <数据集>_products/doctor/）")
    ap.add_argument("--fail-on-warn", action="store_true", help="有 WARN 也返回退出码 1（CI 用）")
    ap.add_argument("--fail-on-fail", action="store_true", help="有 FAIL 返回退出码 1（默认也返回 1）")
    args = ap.parse_args()

    cfg: dict = {}
    p = Path(args.config)
    if p.is_file():
        try:
            import yaml
            with open(p, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:  # noqa: BLE001
            cfg = {}

    cmd = resolve_doctor(args, cfg)
    if cmd is None:
        print(INSTALL_HINT)
        return 1

    rc = 0
    for s in args.input:
        rc |= run_one(Path(s), cmd, args, cfg)
    return rc


if __name__ == "__main__":
    sys.exit(main())
