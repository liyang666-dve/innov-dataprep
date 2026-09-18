#!/usr/bin/env python3
"""阈值定标：用你自己的数据算出 03 的 qc.* 建议值（只读，不改数据、不改配置）。

为什么需要：03 里的关节/尖峰阈值原来是拍脑袋的（joint_jump_rad=0.8、stuck_s=0.4），
在真实 ARX 数据上实测 max 跳变 0.68 rad、最长零方差 18.97 s —— 照旧值判定会大量误杀。
本工具扫若干数据集，输出每维分位数 + 一份可直接粘贴进 config.yaml 的 qc 建议块。

铁律：只读数据；只写报告到 --out；**不修改 config.yaml**（人工确认后再粘贴）。

用法:
    python3 tools/calibrate_qc.py --input <数据集> [--input ...]
    python3 tools/calibrate_qc.py --dir <批次目录>          # 扫一层子目录里所有 v2.1/v3.0
        [--out 目录] [--max-episodes 30] [--config config.yaml]
产物: <out>/calibrate_report.md + qc_suggested.yaml + calibrate.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipe.lib import dataset_io, report  # noqa: E402

PREFIXES = (("observation.state", "state"), ("action", "action"))


def _episode_frames(ds: Path, max_eps: int):
    """产出 (episode_index, 已展开数组列的 DataFrame)；v2.1 与 v3.0 通用。"""
    import pandas as pd

    kind, _ = dataset_io.detect_dataset(ds)
    if kind == "v3.0":
        for i, (ep, df) in enumerate(dataset_io.iter_v3_episodes(ds)):
            if max_eps and i >= max_eps:
                break
            yield ep, df
        return
    for i, p in enumerate(dataset_io.discover_episodes(ds)):
        if max_eps and i >= max_eps:
            break
        yield dataset_io.episode_index(p), dataset_io.expand_array_features(pd.read_parquet(p))


def _pct(a: np.ndarray, q: float) -> float:
    return float(np.nanpercentile(a, q)) if a.size else float("nan")


def scan(ds: Path, max_eps: int) -> dict[str, Any]:
    """单数据集扫描：逐维分位数 + 全局分布。"""
    per: dict[str, list[np.ndarray]] = {tag: [] for _, tag in PREFIXES}
    steps: dict[str, list[np.ndarray]] = {tag: [] for _, tag in PREFIXES}
    stds: dict[str, list[np.ndarray]] = {tag: [] for _, tag in PREFIXES}
    ap: dict[str, list[np.ndarray]] = {tag: [] for _, tag in PREFIXES}
    stuck_run_max = {"state": 0.0, "action": 0.0}
    idle: dict[str, list[float]] = {"state": [], "action": []}
    durs: list[float] = []
    n_ep = 0
    fps = 30.0
    try:
        fps = float(dataset_io.read_meta(ds)["info"].get("fps") or 30.0)
    except Exception:  # noqa: BLE001
        pass
    for _ep, df in _episode_frames(ds, max_eps):
        n_ep += 1
        dt = 1.0 / fps if fps else 0.033
        if "timestamp" in df.columns and len(df) > 1:
            t = df["timestamp"].to_numpy(dtype=np.float64)
            durs.append(float(t[-1] - t[0]))
        else:
            durs.append(len(df) * dt)
        for prefix, tag in PREFIXES:
            cols = [c for c in df.columns if c.startswith(prefix + ".")]
            if not cols:
                continue
            X = df[cols].to_numpy(dtype=np.float64)
            X = X[np.isfinite(X).all(axis=1)]
            if X.shape[0] < 2:
                continue
            with np.errstate(all="ignore"):
                stds[tag].append(np.nanstd(X, axis=0))
                ap[tag].append(np.nanmax(np.abs(X), axis=0))
                D = np.diff(X, axis=0)
                steps[tag].append(np.abs(D))
                eq = np.abs(D) < 1e-12
                run = 0
                for c in range(eq.shape[1]):
                    cnt = mx = 0
                    for v in eq[:, c]:
                        cnt = cnt + 1 if v else 0
                        mx = max(mx, cnt)
                    run = max(run, mx)
                stuck_run_max[tag] = max(stuck_run_max[tag], run * dt)
                idle[tag].append(float((np.linalg.norm(D, axis=1) < 1e-3).mean()))
    out: dict[str, Any] = {"dataset": ds.name, "path": str(ds), "kind": dataset_io.detect_dataset(ds)[0],
                           "n_episodes_scanned": n_ep, "fps": fps}
    for _prefix, tag in PREFIXES:
        if not stds[tag]:
            continue
        S = np.vstack(stds[tag])
        A = np.vstack(ap[tag])
        D = np.vstack(steps[tag])
        out[tag] = {
            "n_dims": int(S.shape[1]),
            "std_min_per_dim": [round(float(x), 4) for x in np.nanmin(S, axis=0)],
            "max_abs_per_dim": [round(float(x), 3) for x in np.nanmax(A, axis=0)],
            "step_p999_per_dim": [round(float(x), 3) for x in np.nanpercentile(D, 99.9, axis=0)],
            "step_max": round(float(np.nanmax(D)), 3),
            "longest_stuck_s": round(stuck_run_max[tag], 3),
            "idle_ratio_p50": round(float(np.median(idle[tag])), 3) if idle[tag] else None,
            "idle_ratio_max": round(float(np.max(idle[tag])), 3) if idle[tag] else None,
        }
    if durs:
        d = np.asarray(durs, dtype=np.float64)
        out["duration_s"] = {"min": round(float(d.min()), 2), "p50": round(float(np.median(d)), 2),
                             "max": round(float(d.max()), 2)}
    return out


def suggest(scans: list[dict]) -> dict[str, Any]:
    """按扫描结果给一份保守建议：以**观测最大值**为下界（宁可放过，不可误杀）。"""
    st = [s["state"] for s in scans if s.get("state")]
    ac = [s["action"] for s in scans if s.get("action")]
    sug: dict[str, Any] = {}
    if st:
        max_abs = max(max(r["max_abs_per_dim"]) for r in st)
        sug["joint_limits_rad"] = round(max(3.3, max_abs * 1.2), 1)
        sug["joint_jump_rad"] = round(max(max(r["step_max"] for r in st) * 1.5,
                                          _p99max(st, "step_p999_per_dim") * 2.0), 2)
        sug["stuck_s"] = round(max(r["longest_stuck_s"] for r in st) * 1.5, 1)
    if ac:
        sug["action_spike_min_abs"] = round(_p99max(ac, "step_p999_per_dim") * 2.0, 2)
    stds_min = [min(r["std_min_per_dim"]) for r in (st + ac) if r.get("std_min_per_dim")]
    nz = [v for v in stds_min if v > 0]
    if nz:
        sug["zero_var_eps"] = float(f"{min(nz) / 10:.1e}")
    idle_max = max([r.get("idle_ratio_max") or 0 for r in (st + ac)] or [0])
    sug["idle_ratio_warn"] = round(min(0.98, max(0.90, idle_max + 0.1)), 2)
    dur = [s["duration_s"] for s in scans if s.get("duration_s")]
    if dur:
        sug["min_duration_s"] = round(max(1.0, min(d["min"] for d in dur) * 0.8), 1)
        sug["max_duration_s"] = round(max(d["max"] for d in dur) * 1.2, 1)
    return sug


def _p99max(rows: list[dict], key: str) -> float:
    vals = [max(r[key]) for r in rows if r.get(key)]
    return float(max(vals)) if vals else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description="用你的数据给 03 的 qc.* 阈值定标（只读）")
    ap.add_argument("--input", action="append", default=[])
    ap.add_argument("--dir", default=None, help="批次目录：扫一层子目录里所有数据集")
    ap.add_argument("--out", default="qc_calibration")
    ap.add_argument("--max-episodes", type=int, default=30, help="每个数据集最多扫多少集（0=全部）")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    targets: list[Path] = [Path(x) for x in args.input]
    if args.dir:
        root = Path(args.dir).expanduser()
        for child in sorted(root.iterdir()):
            if child.is_dir() and not child.name.startswith(("_", ".")):
                if dataset_io.detect_dataset(child)[0] in ("v2.1", "v3.0"):
                    targets.append(child)
    if not targets:
        print("[ERROR] 请给 --input <数据集> 或 --dir <批次目录>")
        return 1

    scans = []
    for ds in targets:
        kind, why = dataset_io.detect_dataset(ds)
        if kind not in ("v2.1", "v3.0"):
            print(f"[跳过] {ds.name}: {why}")
            continue
        r = scan(ds, args.max_episodes)
        scans.append(r)
        print(f"[扫描] {ds.name} ({r['kind']}, {r['n_episodes_scanned']} 集)")
    if not scans:
        print("[ERROR] 没有可扫描的数据集")
        return 1

    sug = suggest(scans)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    report.write_json(out_root / "calibrate.json", {"scans": scans, "suggested_qc": sug})
    yaml_lines = ["# 由 tools/calibrate_qc.py 生成 —— 人工确认后再粘贴进 config.yaml 的 qc: 段",
                  "# 原则：以观测最大值为下界（宁可放过，不可误杀）；粘贴后可用 --joints 打开关节类判定", "",
                  "qc:"]
    for k, v in sug.items():
        yaml_lines.append(f"  {k}: {v}")
    (out_root / "qc_suggested.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    md = ["# qc 阈值定标报告", "",
          f"- 扫描数据集: {len(scans)} 个，每个最多 {args.max_episodes or '全部'} 集",
          f"- 产物: `calibrate.json` / `qc_suggested.yaml`", "",
          "## 建议值（可直接粘贴 config.yaml）", "", "```yaml"] + yaml_lines[3:] + ["```", "",
          "## 逐数据集观测", "",
          "| 数据集 | 版本 | 集数 | state 最大角(rad) | 最大跳变(rad) | 最长零方差(s) | 静止帧占比 p50/max | 时长 min/p50/max |",
          "|---|---|---|---|---|---|---|---|"]
    for s in scans:
        st = s.get("state") or {}
        ac = s.get("action") or {}
        d = s.get("duration_s") or {}
        md.append("| {n} | {k} | {e} | {a} | {j} | {s_} | {i}/{im} | {d1}/{d2}/{d3} |".format(
            n=s["dataset"], k=s["kind"], e=s["n_episodes_scanned"],
            a=max(st.get("max_abs_per_dim") or [0]), j=max(ac.get("step_max") or [0], st.get("step_max") or [0]),
            s_=max(st.get("longest_stuck_s") or 0, ac.get("longest_stuck_s") or 0),
            i=st.get("idle_ratio_p50"), im=st.get("idle_ratio_max"),
            d1=d.get("min"), d2=d.get("p50"), d3=d.get("max")))
    md += ["", "## 逐维细节（state / action 最大绝对跳变 p99.9）", ""]
    for s in scans:
        for tag in ("state", "action"):
            if s.get(tag):
                md.append(f"- **{s['dataset']} / {tag}**: max|值|={s[tag]['max_abs_per_dim']}")
                md.append(f"  - |Δ| p99.9={s[tag]['step_p999_per_dim']}  "
                          f"逐维最小 std={s[tag]['std_min_per_dim']}")
    md += ["", "## 怎么用", "",
           "1. 把 `qc_suggested.yaml` 里想要的值粘进 config.yaml 的 `qc:` 段；",
           "2. 重跑 `python3 run.py clean <编号>`，确认 exclude 集没有异常增加；",
           "3. 确认无误后再加 `--joints` 打开关节限位/跳变/卡死判定。", ""]
    report.write_md(out_root / "calibrate_report.md", md)
    print(f"[OK] 定标报告 -> {out_root / 'calibrate_report.md'}")
    print("     建议值 -> " + ", ".join(f"{k}={v}" for k, v in sug.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
