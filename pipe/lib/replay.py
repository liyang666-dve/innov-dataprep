#!/usr/bin/env python3
"""07 回放：把任一 episode 的数据(状态/动作标量)生成 Rerun 回放 .rrd。

供 Web「回放」页与「一键动作→回放」使用；只读，不修改数据；重依赖 rerun-sdk(libdav1d/ffmpeg 已内置)。

生成思路：
  - v2.1/v3.0 都先经 dataset_io 归一成 v2.1 风格展开列(observation.state.* / action.*)；
  - 以时间("t")为时间轴，用 send_columns 批量写每条 state/action 标量列成 Rerun 曲线；
  - 浏览器 viewer 通过 ?url=<http .rrd> 按静态文件加载并渲染(已实测可行)。

用法:
    python3 pipe/07_replay.py --input <dataset> --episode <N> --out <file.rrd>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipe.lib import dataset_io  # noqa: E402


def list_episodes(ds: Path) -> list[dict]:
    """本数据集的 episode 列表：{episode, n_rows}（快路径，不读全量数据）。"""
    ds = Path(ds)
    kind, _ = dataset_io.detect_dataset(ds)
    out = []
    if kind == "v3.0":
        n = int(dataset_io.summarize_light(ds).get("n_episodes") or 0)
        out = [{"episode": i, "n_rows": -1} for i in range(n)]
    elif kind == "v2.1":
        for p in dataset_io.discover_episodes(ds):
            ep = dataset_io.episode_index(p)
            try:
                n = len(pd.read_parquet(p, columns=["timestamp"]))
            except Exception:  # noqa: BLE001
                n = -1
            out.append({"episode": ep, "n_rows": n})
    out.sort(key=lambda x: x["episode"])
    return out


def _episode_df(ds: Path, ep: int) -> tuple[str, pd.DataFrame]:
    """返回(数据名, 展开后的 ep df)或抛 FileNotFoundError。"""
    ds = Path(ds)
    kind, _ = dataset_io.detect_dataset(ds)
    if kind == "v3.0":
        for e, df in dataset_io.iter_v3_episodes(ds):
            if e == ep:
                return f"{ds.name}_ep{ep}", df
        raise FileNotFoundError(f"{ds.name} 无 episode {ep}")
    if kind == "v2.1":
        for p in dataset_io.discover_episodes(ds):
            if dataset_io.episode_index(p) == ep:
                df = pd.read_parquet(p)
                return f"{ds.name}_ep{ep}", dataset_io._v3_expand_v21(df)
        raise FileNotFoundError(f"{ds.name} 无 episode {ep}")
    raise ValueError(f"{ds.name} 不是 v2.1/v3.0 数据集")


def _timeline(df: pd.DataFrame, fps: float) -> np.ndarray:
    if "timestamp" in df.columns:
        t = df["timestamp"].to_numpy(dtype=np.float64)
        if np.isfinite(t).any():
            t = np.where(np.isfinite(t), t, 0.0)
            return t - float(t.min())  # 相对到 0，避免绝对大数作时间轴
    return np.arange(len(df), dtype=np.float64) / float(fps)


def build_rrd(ds: Path, ep: int, out: Path, fps: float = 30.0) -> Path:
    """把 episode 的状态/动作标量写成 Rerun .rrd；返回输出路径。"""
    try:
        import rerun as rr
        from rerun import components as C
    except ImportError:  # pragma: no cover
        raise RuntimeError("缺少 rerun-sdk：python -m pip install rerun-sdk")  # noqa: TRY003

    name, df = _episode_df(ds, ep)
    ts = _timeline(df, fps)
    rr.init(name, spawn=False)
    col = rr.TimeColumn("t", duration=ts.astype(np.float64))

    state_cols = [c for c in df.columns if c.startswith("observation.state.")]
    act_cols = [c for c in df.columns if c.startswith("action.")]
    for cols, grp in ((state_cols, "state"), (act_cols, "action")):
        for c in cols:
            try:
                vals = df[c].to_numpy(dtype=np.float64)
            except Exception:  # noqa: BLE001
                continue
            feature = c.split(".", 2)[2] if grp == "state" else c[len("action."):]
            rr.send_columns(f"{grp}/{feature}", [col],
                            [rr.ComponentColumn("Scalar", C.ScalarBatch(vals))])
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rr.save(str(out))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="v2.1/v3.0 数据集路径")
    ap.add_argument("--episode", type=int, required=True, help="要回放的 episode 编号")
    ap.add_argument("--out", required=True, help="输出 .rrd 路径")
    ap.add_argument("--fps", type=float, default=30.0)
    a = ap.parse_args(argv)
    out = build_rrd(Path(a.input), a.episode, Path(a.out), a.fps)
    print(f"[OK] 已生成 {out} ({out.stat().st_size} bytes) ep{a.episode}")
    return 0


if __name__ == "__main__":
    sys.exit(main())