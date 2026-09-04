"""Web 后端薄封装：复用 run.py / pipe.lib 的纯函数，向路由提供 HTTP 可调聚合。

不做任何业务重实现；所有动作执行走 run.execute_action，产物读取走既有函数。
"""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from run import (  # noqa: E402
    load_config,
    scan_roots,
    load_state,
    save_state,
    mark,
    state_path,
    ops_file,
    execute_action,
    action_kind_hint,
)

from pipe.lib import dataset_io  # noqa: E402
from pipe.lib import replay  # noqa: E402

STEPS = ["inspect", "timestamps", "clean", "merge", "convert", "record", "annotate"]
STEP_LABELS = {
    "inspect": "盘点",
    "timestamps": "时间戳",
    "clean": "清洗",
    "merge": "合并",
    "convert": "转换",
    "record": "登记",
    "annotate": "标注",
}


def _config_state() -> tuple[dict, dict, Path]:
    cfg = load_config()
    sf = state_path(cfg, None)
    return cfg, load_state(sf), sf


def _fmt_status(st: dict, path: str) -> list[dict]:
    s = st.get(path, {})
    out = []
    for step in STEPS:
        if s.get(step):
            out.append({"step": step, "label": STEP_LABELS.get(step, step), "ts": s[step]})
    return out


def get_overview() -> dict:
    """批次总览：分组(采集/输出) + 每批版本/集数/状态。"""
    cfg, st, _ = _config_state()
    items = scan_roots(cfg, None)
    rows = []
    for i, info in enumerate(items, 1):
        path = info["path"]
        done = {x["step"] for x in _fmt_status(st, path)}
        rows.append({
            "idx": i,
            "name": info.get("name"),
            "path": path,
            "group": info.get("group", "批次"),
            "kind": info.get("kind"),
            "reason": info.get("reason", ""),
            "n_episodes": info.get("n_episodes"),
            "fps": info.get("fps"),
            "robot_type": info.get("robot_type"),
            "done": sorted(done),
        })
    return {"groups": ["采集", "输出"], "rows": rows}


def _discover_artifacts(path: str) -> dict:
    """查该数据集的清洗产物（清洗汇总含 n_keep/n_exclude）。
    新布局 <名字>_products/clean 优先、回退旧平铺 <名字>_clean（按文件实际存在判定）。"""
    p = Path(path)
    out = {"summary_json": None, "disposition": None, "clean_dir": None}
    sj = dataset_io.stage_file(p, "clean", "summary.json")
    if sj:
        try:
            out["summary_json"] = json.loads(sj.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
        out["clean_dir"] = str(sj.parent)
    disp = dataset_io.stage_file(p, "clean", "episode_disposition.csv")
    if disp:
        out["disposition"] = full_disposition(disp)
        if not out["clean_dir"]:
            out["clean_dir"] = str(disp.parent)
    return out


def full_disposition(csv_path: Path) -> list[dict]:
    """读 episode_disposition.csv 成列表（保留全部字段，供明细表渲染）。"""
    if not Path(csv_path).is_file():
        return []
    try:
        # utf-8-sig：兼容带 BOM 的 CSV（否则首列名变 "\ufeffepisode" 取不到值）
        with open(csv_path, encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    except Exception:  # noqa: BLE001
        return []


def get_batch(path: str) -> dict:
    """批次详情：摘要(轻量)+状态+清洗产物(disposition/summary)+操作链。"""
    cfg, st, sf = _config_state()
    p = Path(path).expanduser().resolve()
    info = dataset_io.summarize_light(p)
    detail = {
        "path": str(p),
        "name": info.get("name"),
        "kind": info.get("kind"),
        "reason": info.get("reason", ""),
        "light": {k: info.get(k) for k in
                  ("n_episodes", "fps", "robot_type", "n_frames", "duration_h",
                   "min_date", "max_date", "task_names", "cameras") if k in info},
        "status": _fmt_status(st, str(p)),
        "artifacts": _discover_artifacts(str(p)),
        "ops": read_ops(sf, str(p)),
    }
    return detail


def read_ops(sf: Path, path: str) -> list[dict]:
    """操作留痕：读取 .dataprep_ops.jsonl，过滤属于该批次的记录。"""
    f = ops_file(sf)
    rows = []
    if not f.is_file():
        return rows
    try:
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if r.get("path") == path or r.get("name") == Path(path).name:
                rows.append(r)
    except Exception:  # noqa: BLE001
        pass
    return rows


def run_action(action: str, paths: list[str], opts: dict | None = None,
               emit=None) -> dict:
    """执行一键动作（自由编排）。emit(callable) 逐行推日志给 SSE。"""
    cfg, st, sf = _config_state()
    opts = opts or {}
    # convert 的 06_convert.py 与 record 的确认提示若不带 --yes 会走 input() 等终端确认，
    # Web 子进程无终端会卡死 -> 自动带 --yes
    if action in ("convert", "record") and opts.get("yes") is None:
        opts["yes"] = True
    selected = [dataset_io.summarize_light(Path(p).expanduser().resolve()) for p in paths]
    rc = execute_action(action, selected, cfg, st, sf, opts, out_stream=emit)
    return {"rc": rc, "action": action}


def action_options() -> list[dict]:
    """供前端动作页渲染。'replay' 为跳转伪动作（前端拦截，不进入 execute_action）。"""
    return [
        {"value": "inspect", "label": "盘点(01)", "need2": False},
        {"value": "timestamps", "label": "时间戳审计(02)", "need2": False},
        {"value": "clean", "label": "清洗质检(03)", "need2": False},
        {"value": "annotate", "label": "标注(VLM 09)", "need2": False},
        {"value": "replay", "label": "视频复核 · 打开", "need2": False},
        {"value": "merge", "label": "合并(05)", "need2": True},
        {"value": "convert", "label": "转换v2.1→v3.0(06)", "need2": False},
        {"value": "record", "label": "登记台账", "need2": False},
    ]


def _split_reasons(cell) -> list[str]:
    if not cell:
        return []
    cell = str(cell).strip()
    if cell in ("-", "", "None"):
        return []
    return [x.strip() for x in cell.split("|") if x.strip()]


def suggest(reason: str) -> str:
    """把质检发现的问题映射成该怎么做（处理建议）。"""
    r = (reason or "").lower()
    pairs = [
        ("状态-动作特征不一致", "特征列不匹配，建议排除该集（重录）"),
        ("时长过短", "采集太短，建议排除并重录"),
        ("时长偏长", "超时太长，建议拆分成多段"),
        ("实际帧率", "帧率偏离标称，建议重采并检查采集端"),
        ("丢帧", "存在丢帧，建议重采/检查采集稳定性"),
        ("时间戳回退", "时间戳错乱，建议重采"),
        ("重复时间戳", "有重复帧，建议复核"),
        ("nan", "含 NaN/Inf 数据，建议排除"),
        ("关节超限位", "关节越界，需检查机械/安全限制"),
        ("关节跳变", "关节异常跳变，建议排除并检查采集"),
        ("关节卡死", "关节卡死，建议排除并检查硬件"),
        ("模糊", "画面模糊，建议重录该机位"),
        ("视频缺失", "视频缺失，建议排除该集"),
        ("视频帧数不符", "音视频不同步，建议排除"),
        ("视频帧数差1", "仅差一帧，可复核后保留"),
    ]
    for kw, s in pairs:
        if kw in r:
            return s
    return "建议排除该集并复核"


def get_qc() -> dict:
    """质检与处理建议：汇总每个已清洗(v2.1)数据集的逐集 verdict + 问题 + 处理建议。"""
    cfg, st, _ = _config_state()
    items = scan_roots(cfg, None)
    batches = []
    for info in items:
        if info.get("kind") != "v2.1":
            continue
        art = _discover_artifacts(info["path"])
        disp = art["disposition"] or []
        if not disp:
            continue  # 未执行清洗质检(03)的批次不在此页展示
        per_ep, n_ex, n_warn = [], 0, 0
        for row in disp:
            verdict = (row.get("verdict") or "keep").strip()
            rex = _split_reasons(row.get("reasons_exclude"))
            rwn = _split_reasons(row.get("reasons_warn"))
            handling = ""
            if verdict == "exclude":
                handling = suggest(rex[0]) if rex else "建议排除"
            elif rwn:
                handling = "可保留（有警告）"
            if verdict == "exclude":
                n_ex += 1
            if rwn:
                n_warn += 1
            per_ep.append({
                "episode": row.get("episode"),
                "n_rows": row.get("n_rows"),
                "duration_s": row.get("duration_s"),
                "verdict": verdict,
                "reasons_exclude": rex,
                "reasons_warn": rwn,
                "handling": handling,
            })
        sj = art["summary_json"] or {}
        n_keep = sj.get("n_keep")
        if n_keep is None:
            n_keep = sum(1 for p in per_ep if p["verdict"] == "keep")
        batches.append({
            "name": info.get("name"),
            "path": info["path"],
            "n_episodes": info.get("n_episodes"),
            "clean_dir": art["clean_dir"],
            "n_keep": n_keep,
            "n_exclude": n_ex,
            "n_warn": n_warn,
            "excluded_episodes": sj.get("excluded_episodes") or [
                p["episode"] for p in per_ep if p["verdict"] == "exclude"],
            "per_episode": per_ep,
        })
    return {"batches": batches}


def get_deliver() -> dict:
    """交付页候选源：批次自我识别(kind/已完成步) + 前置校验(可交付?) + 历史交付次数。"""
    cfg, st, sf = _config_state()
    items = scan_roots(cfg, None)
    rows = []
    for info in items:
        path = info["path"]
        kind = info.get("kind")
        done = {x["step"] for x in _fmt_status(st, path)}
        delivers = [o for o in read_ops(sf, path) if o.get("action") == "deliver"]
        if kind not in ("v2.1", "v3.0"):
            level, reason = "block", f"非 v2.1/v3.0 数据集：{info.get('reason') or '缺 meta/info.json'}"
        elif "clean" not in done:
            level, reason = "warn", "尚未清洗质检(03)，交付的是原始数据，建议先清洗再交付"
        else:
            level, reason = "ok", "可交付"
        rows.append({
            "name": info.get("name"),
            "path": path,
            "group": info.get("group", "批次"),
            "kind": kind,
            "done": sorted(done),
            "n_episodes": info.get("n_episodes"),
            "deliver_count": len(delivers),
            "last_deliver": delivers[-1]["note"] if delivers else None,
            "level": level,
            "reason": reason,
        })
    return {"rows": rows}


# ---------------------------------------------------------------- 视频复核(review)
def _clean_disposition_map(path: str) -> dict[str, dict]:
    """该数据集若已清洗，返回 {episode(str): disposition 行}；否则 {}。"""
    art = _discover_artifacts(path)
    disp = art["disposition"] or []
    return {str(r.get("episode")): r for r in disp}


def get_review_episodes(path: str) -> dict:
    """复核页 episode 列表：集号/帧数/时长/质检结论(verdict+原因+建议)/视频机位。"""
    ds = Path(path).expanduser().resolve()
    kind, reason = dataset_io.detect_dataset(ds)
    if kind not in ("v2.1", "v3.0"):
        return {"path": str(ds), "kind": kind, "error": f"不是 v2.1/v3.0 数据集：{reason}", "episodes": []}
    meta = dataset_io.read_meta(ds)
    cams = dataset_io.camera_layout(ds, meta.get("info") or {})
    cam_keys = sorted(cams.keys())
    disp = _clean_disposition_map(str(ds))

    eps: list[dict] = []
    if kind == "v2.1":
        for p in dataset_io.discover_episodes(ds):
            ep = dataset_io.episode_index(p)
            try:
                n = len(pd.read_parquet(p, columns=["timestamp"]))
            except Exception:  # noqa: BLE001
                n = -1
            eps.append({"episode": ep, "n_rows": n})
    else:  # v3.0：从 meta/episodes 读每集长度与每机位起止时间
        emeta = dataset_io._v3_episodes_meta(ds)
        for _, row in emeta.iterrows():
            ep = int(row["episode_index"])
            win = {}
            for cam in cam_keys:
                ft = row.get(f"videos/{cam}/from_timestamp")
                tt = row.get(f"videos/{cam}/to_timestamp")
                if ft is not None and tt is not None and not pd.isna(ft) and not pd.isna(tt):
                    win[cam] = [float(ft), float(tt)]
            eps.append({"episode": ep, "n_rows": int(row.get("length") or 0), "window": win})
        eps.sort(key=lambda x: x["episode"])

    fps = float((meta.get("info") or {}).get("fps") or 30.0)
    for e in eps:
        key = str(e["episode"])
        d = disp.get(key)
        if d:
            verdict = (d.get("verdict") or "keep").strip()
            rex = _split_reasons(d.get("reasons_exclude"))
            rwn = _split_reasons(d.get("reasons_warn"))
            e["verdict"] = verdict
            e["reasons"] = rex + [f"⚠ {w}" for w in rwn]
            if verdict == "exclude":
                e["handling"] = suggest(rex[0]) if rex else "建议排除"
            elif rwn:
                e["handling"] = "可保留（有警告）"
            else:
                e["handling"] = "保留"
        else:
            e["verdict"] = None
            e["reasons"] = []
            e["handling"] = "未清洗（建议先跑清洗质检 03）"
        if e.get("n_rows", -1) > 0 and fps > 0:
            e["duration_s"] = round(e["n_rows"] / fps, 1)
    return {"path": str(ds), "name": ds.name, "kind": kind, "fps": fps,
            "cameras": cam_keys, "cleaned": bool(disp), "episodes": eps}


def review_video(path: str, ep: int, cam: str | None) -> dict:
    """定位某集某机位的 mp4 文件。v2.1 直接是独立文件；v3.0 是机位大 mp4 + 起止时间窗。"""
    ds = Path(path).expanduser().resolve()
    kind, reason = dataset_io.detect_dataset(ds)
    if kind not in ("v2.1", "v3.0"):
        return {"error": f"不是 v2.1/v3.0 数据集：{reason}"}
    meta = dataset_io.read_meta(ds)
    cams = sorted(dataset_io.camera_layout(ds, meta.get("info") or {}).keys())
    if not cams:
        return {"error": "未发现任何机位视频"}
    cam = cam or cams[0]
    if cam not in cams:
        return {"error": f"机位 {cam} 不存在，可选：{cams}"}

    if kind == "v2.1":
        chunk = f"chunk-{ep // 1000:03d}"
        f = ds / "videos" / chunk / cam / f"episode_{ep:06d}.mp4"
        if not f.is_file():
            cands = list((ds / "videos").glob(f"*/{cam}/episode_{ep:06d}.mp4"))
            f = cands[0] if cands else f
        if not f.is_file():
            return {"error": f"未找到 {cam} ep{ep} 的视频文件"}
        return {"file": str(f), "start": 0.0, "end": None, "cameras": cams, "cam": cam}

    # v3.0：机位大 mp4 + 该集时间窗
    mp4 = dataset_io._v3_camera_mp4(ds, {cam: {}}).get(cam)
    if not mp4 or not Path(mp4).is_file():
        return {"error": f"未找到 {cam} 的机位视频"}
    emeta = dataset_io._v3_episodes_meta(ds)
    row = emeta[emeta["episode_index"] == ep]
    start, end = 0.0, None
    if not row.empty:
        ft = row.iloc[0].get(f"videos/{cam}/from_timestamp")
        tt = row.iloc[0].get(f"videos/{cam}/to_timestamp")
        if ft is not None and not pd.isna(ft):
            start = float(ft)
        if tt is not None and not pd.isna(tt):
            end = float(tt)
    return {"file": str(mp4), "start": start, "end": end, "cameras": cams, "cam": cam}


# ---------------------------------------------------------------- 回放(rerun)
_REPLAY_PORT = 9888
_REPLAY_VIEWER = {"pid": None, "port": _REPLAY_PORT}

def _state_dir() -> Path:
    _, _, sf = _config_state()
    return sf.parent if sf.parent.is_dir() else Path("/tmp")


def replay_cache_dir() -> Path:
    d = _state_dir() / ".dataprep_replay"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _rrd_path(path: str, ep: int) -> Path:
    return replay_cache_dir() / f"{Path(path).name}_ep{ep}.rrd"


def _pid_alive(pid: int) -> bool:
    if not pid:
        return False
    try:
        import os
        os.kill(pid, 0)
        return True
    except Exception:  # noqa: BLE001
        return False


def ensure_viewer() -> str:
    """确保 Rerun web-viewer 子进程在跑，返回其基址 URL。"""
    import subprocess as sp
    port = _REPLAY_VIEWER["port"]
    if not _pid_alive(_REPLAY_VIEWER["pid"]):
        code = (f"import rerun,time\n"
                f"rerun.serve_web_viewer(web_port={port}, open_browser=False)\n"
                f"import time;time.sleep(10**7)")
        p = sp.Popen([sys.executable, "-c", code],
                     stdout=sp.DEVNULL, stderr=sp.DEVNULL, start_new_session=True)
        _REPLAY_VIEWER["pid"] = p.pid
    return f"http://127.0.0.1:{port}"


def get_replay_sources() -> dict:
    """回放源：v2.1/v3.0 数据集 + 简况。"""
    ov = get_overview()
    rows = [r for r in ov["rows"] if r["kind"] in ("v2.1", "v3.0")]
    return {"rows": rows}


def get_replay_episodes(path: str) -> dict:
    ds = Path(path).expanduser().resolve()
    return {"path": str(ds), "episodes": replay.list_episodes(ds)}


def rrd_path(path: str, ep: int) -> Path:
    return _rrd_path(path, ep)


def build_replay(path: str, ep: int, emit=None) -> int:
    """后台生成 .rrd（走 SSE 推日志），成功后留痕 action=replay。"""
    say = emit or (lambda s: print(s))
    ds = Path(path).expanduser().resolve()
    out = _rrd_path(str(ds), ep)
    try:
        say(f"==> 生成回放 {ds.name} ep{ep} ...")
        res = replay.build_rrd(ds, ep, out)
        say(f"[OK] {res.name} 已生成（{res.stat().st_size} bytes）")
    except Exception as e:  # noqa: BLE001
        say(f"[ERROR] 回放生成失败：{e}")
        say("[exit]")
        return 1
    try:
        cfg, st, sf = _config_state()
        p = ops_file(sf)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "time": datetime.now().isoformat(timespec="seconds"),
                "action": "replay",
                "name": ds.name, "path": str(ds),
                "kind": "replay",
                "note": f"生成回放 .rrd ep{ep}",
            }, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001
        pass
    say("[exit]")
    return 0


def get_ledger() -> dict:
    """台账页：读 config paths.ledger 的 data_catalog.csv。"""
    cfg, _, _ = _config_state()
    ledger = Path(cfg.get("paths", {}).get("ledger") or "data_catalog.csv").expanduser()
    cols, rows = [], []
    if ledger.is_file():
        try:
            with open(ledger, encoding="utf-8") as f:
                dr = csv.DictReader(f)
                cols = dr.fieldnames or []
                rows = list(dr)
        except Exception:  # noqa: BLE001
            pass
    merged = ledger.parent / "data_catalog_merged.csv"
    return {
        "path": str(ledger),
        "cols": cols,
        "rows": rows,
        "merged_exists": merged.is_file(),
        "merged_path": str(merged) if merged.is_file() else None,
    }