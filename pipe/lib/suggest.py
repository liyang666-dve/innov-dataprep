"""VLM 标注：用 OpenAI 兼容接口(DeepSeek/通义等) 逐集生成数据质量评分+处理建议。

只读，不修改数据；未启用/缺端点/缺 Key 时优雅拦截（返回 reason），绝不硬崩。
产物写 <dataset>_annotation/suggestions.jsonl + summary.json，供 Web「一键动作 → 标注」、
CLI 与后续人工复核使用。

低成本的文本方案：把每集的 joint/action 曲线统计 + 元信息拼成 prompt，交给 VLM 推断
任务完成度与机械层面风险（静止卡死、突跳、幅度异常等），输出 {score, issues, suggestion}。
不改动既有 QC；若要叠加视频帧，可在此基础上逐步加图片输入。
"""
from __future__ import annotations

import json
import os
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import replay

SYSTEM_PROMPT = (
    "你是机器人数据采集质量评审。根据给出的某一条 demonstration 的关节状态/动作曲线统计，"
    "判断其作为机械臂操作（如抓取/放置）训练数据的质量。输出严格 JSON，不要任何多余文字："
    '{"score": 0到100的整数, "issues": ["问题数组"], "suggestion": "一句话处理建议"}。'
    "评分基准：曲线连续平滑、末端执行器有实际运动、无明显突跳/长时间静止=高分；"
    "末端全程无运动/长时间零方差(疑似卡死)、突跳超限、幅度异常=低分并明确给出建议(排除/复核)。"
)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """读 config.yaml（缺依赖或解析失败返回 {}，绝不抛）。"""
    try:
        import yaml
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:  # noqa: BLE001
        return {}


def annotate_config(cfg: dict[str, Any] | None) -> dict[str, Any]:
    """校验 config.yaml 的 annotate 段。返回 {ok, reason, conf}。

    任一必要条件缺失即 ok=False 并给出 reason；这是『无端点拦截』与『config 校验』的入口。
    """
    a = (cfg or {}).get("annotate") or {}
    base = (a.get("base_url") or "").strip()
    model = (a.get("model") or "").strip()
    env = (a.get("api_key_env") or "").strip()
    if not a.get("enabled"):
        return {"ok": False, "reason": "标注未启用（config.yaml: annotate.enabled=false）", "conf": a}
    if not base:
        return {"ok": False, "reason": "缺 annotate.base_url（OpenAI 兼容地址，示例 https://api.deepseek.com/v1）", "conf": a}
    if not model:
        return {"ok": False, "reason": "缺 annotate.model（示例 deepseek-chat）", "conf": a}
    if not env:
        return {"ok": False, "reason": "缺 annotate.api_key_env（环境变量名，存放 API Key）", "conf": a}
    if not os.environ.get(env):
        return {"ok": False, "reason": f"环境变量 {env} 未设置（API Key 缺失）", "conf": a}
    return {"ok": True, "reason": "", "conf": a}


def _duration_s(df: pd.DataFrame, fps: float = 30.0) -> float:
    if "timestamp" in df.columns:
        t = df["timestamp"].to_numpy(dtype=np.float64)
        t = t[np.isfinite(t)]
        if t.size >= 2:
            return float(t[-1] - t[0])
    return float(len(df)) / fps


def _feature_stats(df: pd.DataFrame, prefix: str, label_fn) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cols = [c for c in df.columns if c.startswith(prefix)]
    for c in cols[:16]:
        try:
            v = df[c].to_numpy(dtype=np.float64)
        except Exception:  # noqa: BLE001
            continue
        v = v[np.isfinite(v)]
        if v.size < 2:
            continue
        jmp = float(np.max(np.abs(np.diff(v)))) if v.size > 1 else 0.0
        rng = float(np.ptp(v))
        out.append({
            "f": label_fn(c),
            "min": round(float(v.min()), 3),
            "max": round(float(v.max()), 3),
            "spread": round(rng, 3),
            "max_jump": round(jmp, 3),
            "const": bool(rng < 1e-6),
        })
    return out


def _summarize(df: pd.DataFrame, ep: int, fps: float = 30.0) -> dict[str, Any]:
    def k(c: str) -> str:
        return c.split(".", 2)[2]
    return {
        "episode": ep,
        "n_rows": len(df),
        "duration_s": round(_duration_s(df, fps), 2),
        "state": _feature_stats(df, "observation.state.", k),
        "action": _feature_stats(df, "action.", lambda c: c[len("action."):]),
    }


def _build_prompt(summary: dict[str, Any]) -> str:
    def block(tag: str, rows: list[dict]) -> str:
        if not rows:
            return f"{tag}: 无"
        lines = [f"{r['f']}: range {r['min']}~{r['max']} spread {r['spread']} "
                 f"max_jump {r['max_jump']} const={r['const']}" for r in rows]
        return tag + ":\n" + "\n".join(lines)

    return (
        f"episode {summary['episode']}：{summary['n_rows']} 帧，"
        f"约 {summary['duration_s']} 秒。\n"
        + block("关节状态", summary["state"])
        + "\n" + block("目标动作", summary["action"])
    )


def _parse_resp(text: str, ep: int) -> dict[str, Any]:
    """宽松解析 VLM 输出 => dict，附 episode；解析失败不致命。"""
    t = (text or "").strip()
    # 去 ```json ``` 围栏
    if t.startswith("```"):
        t = t.split("```", 2)[1] if t.count("```") >= 2 else t.strip("`")
    try:
        obj = json.loads(t)
    except Exception:  # noqa: BLE001
        # 找不到 JSON 就只保留原文，评分留空
        return {"episode": ep, "score": None, "issues": [], "suggestion": text[:200]}
    return {
        "episode": ep,
        "score": obj.get("score"),
        "issues": obj.get("issues") or [],
        "suggestion": obj.get("suggestion") or "",
    }


def call_llm(prompt: str, conf: dict[str, Any], timeout: int = 120) -> str:
    """OpenAI 兼容 /chat/completions（用标准库 urllib，避免新增依赖）。"""
    url = conf["base_url"].rstrip("/") + "/chat/completions"
    payload = {
        "model": conf["model"],
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 800,
    }
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get(conf['api_key_env'], '')}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"VLM HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}")  # noqa: TRY003
    obj = json.loads(body)
    return obj["choices"][0]["message"]["content"]


def annotate(ds: Path, ep: int, conf: dict[str, Any]) -> dict[str, Any]:
    """对单个 episode 生成标注；conf 需已通过 annotate_config（ok=True）。"""
    _, df = replay._episode_df(ds, ep)
    summary = _summarize(df, ep)
    text = call_llm(_build_prompt(summary), conf)
    return _parse_resp(text, ep)


def annotate_batch(ds: Path, out_dir: Path | None, cfg: dict[str, Any], emit=None) -> int:
    """校验 config 后，逐集调用 VLM 生成标注并落盘。返回进程退出码。"""
    say = emit or (lambda s: print(s))
    conf = annotate_config(cfg)
    if not conf["ok"]:
        say(f"[!] 标注被拦截：{conf['reason']}")
        say("[exit]")
        return 1
    ds = Path(ds)
    out_dir = Path(out_dir) if out_dir else replay.dataset_io.new_stage_dir(ds, "annotation")
    out_dir.mkdir(parents=True, exist_ok=True)

    eps = replay.list_episodes(ds)
    if not eps:
        say(f"[!] {ds.name} 无可用 episode")
        say("[exit]")
        return 1

    say(f"==> 标注 {ds.name}：{len(eps)} 集（model={conf['conf']['model']}）")
    results: list[dict[str, Any]] = []
    for i, e in enumerate(eps, 1):
        ep = e["episode"]
        say(f"  [{i}/{len(eps)}] ep{ep} ...")
        try:
            parsed = annotate(ds, ep, conf["conf"])
        except Exception as ex:  # noqa: BLE001
            say(f"    [ERROR] {ex}")
            parsed = {"episode": ep, "score": None, "issues": [], "suggestion": f"调用失败: {ex}"}
        results.append(parsed)

    jl = out_dir / "suggestions.jsonl"
    with jl.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_ok = sum(1 for r in results if r.get("score") is not None)
    kind, _ = replay.dataset_io.detect_dataset(ds) if hasattr(replay, "dataset_io") else ("", "")
    (out_dir / "summary.json").write_text(
        json.dumps({
            "dataset": ds.name, "path": str(ds), "kind": kind,
            "n_episodes": len(eps), "n_scored": n_ok,
            "model": conf["conf"]["model"],
            "suggestions_file": str(jl),
        }, ensure_ascii=False, indent=1), encoding="utf-8")

    say(f"[OK] 标注完成：{n_ok}/{len(eps)} 集有评分；输出 {out_dir}")
    if n_ok < len(eps):
        say(f"     另有 {len(eps) - n_ok} 集调用失败，见 suggestions.jsonl 的 error 字段")
    say("[exit]")
    return 0