"""Flask 本地 Web 界面入口。

启动: python web/app.py  →  http://127.0.0.1:8000 （conda: lerobot_arx_sdk311）

单进程；长任务(动作/传输)在后台线程跑，SSE 实时推日志，页面刷新不影响后台执行。
"""
from __future__ import annotations

import json
import queue
import sys
import threading
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

try:
    from flask import Flask, render_template, request, Response, jsonify, send_file  # noqa: E402
except ModuleNotFoundError as e:
    if getattr(e, "name", None) == "flask":
        print("[ERROR] 缺少 flask（Web 界面依赖，可选）。请先安装：")
        print("        pip install flask          # 基础 Web")
        print("        pip install rerun-sdk      # 回放页/一键回放（不需要可跳过）")
        print("    或在仓库根目录跑 bash setup.sh（自动复用当前 conda 环境并装齐依赖）")
    else:
        print(f"[ERROR] 缺少依赖模块: {e.name}")
    sys.exit(1)

import backend  # noqa: E402
import transfer  # noqa: E402

app = Flask(__name__)

# 任务表: task_id -> {"q": Queue, "thread": Thread}
TASKS: dict[str, dict] = {}


def _emit(q: "queue.Queue"):
    def emit(line):
        q.put(("line", line))
    return emit


def _spawn(fn_builder) -> str:
    """后台跑 fn_builder(emit)；q 收 ('line',t) 与 ('done',rc)。返回 task_id。"""
    q: "queue.Queue" = queue.Queue()
    tid = uuid.uuid4().hex[:12]
    emit = _emit(q)

    def worker():
        rc = fn_builder(emit)
        q.put(("done", rc))

    t = threading.Thread(target=worker, daemon=True)
    TASKS[tid] = {"q": q, "thread": t}
    t.start()
    return tid


def _stream_gen(tid: str):
    q = TASKS[tid]["q"]
    while True:
        try:
            kind, payload = q.get(timeout=15)
        except queue.Empty:
            yield ":\n\n"  # 心跳，保持连接
            continue
        if kind == "done":
            yield f"event: done\ndata: {json.dumps({'rc': payload})}\n\n"
            break
        yield f"data: {json.dumps({'line': payload}, ensure_ascii=False)}\n\n"
    del TASKS[tid]


# ---------------------------------------------------------------- 页面
@app.route("/")
def page_index():
    return render_template("index.html")


@app.route("/batch")
def page_batch():
    return render_template("batch.html", path=request.args.get("path", ""))


@app.route("/action")
def page_action():
    return render_template("action.html")


@app.route("/qc")
def page_qc():
    return render_template("qc.html")


@app.route("/ledger")
def page_ledger():
    return render_template("ledger.html")


@app.route("/deliver")
def page_deliver():
    return render_template("deliver.html")


@app.route("/replay")
def page_replay():
    return render_template("replay.html")


@app.route("/replay/curves")
def page_replay_curves():
    return render_template("replay_curves.html")


# ---------------------------------------------------------------- 数据 API
@app.get("/api/overview")
def api_overview():
    return jsonify(backend.get_overview())


@app.get("/api/batch")
def api_batch():
    path = request.args.get("path", "")
    if not path:
        return jsonify({"error": "缺少 path"}), 400
    return jsonify(backend.get_batch(path))


@app.get("/api/action/options")
def api_action_options():
    return jsonify(backend.action_options())


@app.get("/api/ledger")
def api_ledger():
    return jsonify(backend.get_ledger())


@app.get("/api/qc")
def api_qc():
    return jsonify(backend.get_qc())


# ---------------------------------------------------------------- 交付 rsync
@app.get("/api/deliver")
def api_deliver_list():
    return jsonify(backend.get_deliver())


@app.post("/api/deliver/preview")
def api_deliver_preview():
    body = request.get_json(silent=True) or {}
    target = transfer._target(body.get("host", ""), body.get("user", ""), body.get("dir", ""))
    return jsonify(transfer.preview(body.get("source", ""), target or "", body.get("identity")))


@app.post("/api/deliver")
def api_deliver():
    body = request.get_json(silent=True) or {}
    source = body.get("source", "")
    target = transfer._target(body.get("host", ""), body.get("user", ""), body.get("dir", ""))
    if not source or not target:
        return jsonify({"error": "缺少源(source)或目标(host/user/dir)参数"}), 400
    identity = body.get("identity")
    tid = _spawn(lambda emit: transfer.do_transfer(source, target, identity, emit=emit))
    return jsonify({"task_id": tid})


# ---------------------------------------------------------------- 视频复核(review)
@app.get("/api/review/episodes")
def api_review_episodes():
    path = request.args.get("path", "")
    if not path:
        return jsonify({"error": "缺少 path"}), 400
    return jsonify(backend.get_review_episodes(path))


@app.get("/api/review/video")
def api_review_video():
    """流式提供某集某机位 mp4（send_file 自带 Range，支持进度条拖动）。"""
    path = request.args.get("path", "")
    ep = request.args.get("ep", "")
    cam = request.args.get("cam") or None
    if not path or ep == "":
        return jsonify({"error": "缺少 path/ep"}), 400
    info = backend.review_video(path, int(ep), cam)
    if "error" in info:
        return jsonify(info), 404
    return send_file(info["file"], mimetype="video/mp4", conditional=True)


# ---------------------------------------------------------------- 回放(rerun)
@app.get("/api/replay/sources")
def api_replay_sources():
    return jsonify(backend.get_replay_sources())


@app.get("/api/replay/episodes")
def api_replay_episodes():
    path = request.args.get("path", "")
    if not path:
        return jsonify({"error": "缺少 path"}), 400
    return jsonify(backend.get_replay_episodes(path))


@app.post("/api/replay/build")
def api_replay_build():
    body = request.get_json(silent=True) or {}
    path, ep = body.get("path", ""), body.get("episode")
    if not path or ep is None:
        return jsonify({"error": "缺少 path/episode"}), 400
    # 已缓存则直接返回，避免重复生成
    rrd = backend.rrd_path(path, int(ep))
    if rrd.is_file():
        return jsonify({"cached": True, "task_id": None})
    tid = _spawn(lambda emit: backend.build_replay(path, int(ep), emit=emit))
    return jsonify({"cached": False, "task_id": tid})


@app.get("/api/replay/rrd")
def api_replay_rrd():
    path = request.args.get("path", "")
    ep = request.args.get("ep", "")
    if not path or ep == "":
        return jsonify({"error": "缺少 path/ep"}), 400
    rrd = backend.rrd_path(path, int(ep))
    if not rrd.is_file():
        return jsonify({"error": f"该episode回放未生成（ep{ep}）：需先“构建回放”"}), 404
    resp = Response(rrd.read_bytes(), mimetype="application/octet-stream")
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.get("/api/replay/viewer")
def api_replay_viewer():
    return jsonify({"url": backend.ensure_viewer()})


# ---------------------------------------------------------------- 动作调度(SSE)
@app.post("/api/action")
def api_action():
    body = request.get_json(silent=True) or {}
    action = body.get("action", "")
    paths = body.get("paths", [])
    opts = body.get("opts", {})
    if not action or not paths:
        return jsonify({"error": "缺少 action/paths"}), 400
    tid = _spawn(lambda emit: backend.run_action(action, paths, opts, emit=emit))
    return jsonify({"task_id": tid})


@app.get("/api/stream/<tid>")
def api_stream(tid):
    if tid not in TASKS:
        return jsonify({"error": "任务不存在"}), 404
    return Response(_stream_gen(tid), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/health")
def health():
    return jsonify({"ok": True})


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    try:
        import werkzeug.serving
        werkzeug.serving.run_simple("127.0.0.1", port, app, threaded=True)
    except ImportError:
        app.run(host="127.0.0.1", port=port, threaded=True)