#!/usr/bin/env python3
"""12 汇总 QA 证据：把一次数据集的各步产物合成一份可交付的结论（只读，不改数据）。

为什么需要：01/02/03/07/11 各自产出报告，交付时没人愿意翻 5 个目录。本步骤把它们
合成 `<ds>_products/qa/qa_summary.md + qa_summary.json`，交付包（08）会把它一并带上，
台账（record.py）也会把结论写进 stats 列。

判定口径（三档，与 03 一致）：
  - blocked：doctor FAIL、07 结构校验不通过、或没跑过 03（无质检证据）；
  - review ：有 review 集、doctor WARN、stat 漂移等需要人看；
  - ready  ：以上都没有。

用法:
    python3 pipe/12_qa_report.py --input <数据集> [--config config.yaml]
        [--out 目录] [--require-doctor] [--fail-on ready|review|blocked]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipe.lib import dataset_io, report  # noqa: E402

LEVEL = {"ready": 0, "review": 1, "blocked": 2}


def _read_json(p: Path) -> dict | None:
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _find(ds: Path, stage: str, names: tuple[str, ...]) -> Path | None:
    """某阶段产物文件（新布局 _products/{stage} 优先、回退旧平铺）。"""
    for n in names:
        p = dataset_io.stage_file(ds, stage, n)
        if p:
            return p
    return None


def collect(ds: Path, cfg: dict) -> dict[str, Any]:
    """收集各步产物（缺失即 None，不报错）。"""
    out: dict[str, Any] = {"dataset": ds.name, "path": str(ds)}
    out["clean"] = _read_json(_find(ds, "clean", ("summary.json",)) or Path("/nonexistent"))
    dpath = _find(ds, "doctor", ("doctor.json",))
    out["doctor"] = _read_json(dpath) if dpath else None
    vpath = _find(ds, "verify", ("verify_report.json",))
    out["verify"] = _read_json(vpath) if vpath else None
    tpath = _find(ds, "timestamps", ("timestamps_report.md", "timestamps.json"))
    out["timestamps_report"] = str(tpath) if tpath else None
    out["inspect_report"] = bool(_find(ds, "inspect", ("inspect_report.md", "summary.json")))
    # 交付包（08）：默认在 paths.output 或数据集同级
    packs: list[dict] = []
    for cand_dir in {ds.parent, Path(cfg.get("paths", {}).get("output") or ds.parent).expanduser(),
                     dataset_io.products_dir(ds) / "pack"}:
        try:
            for tar in sorted(cand_dir.glob(f"{ds.name}_delivery.tar.gz*")):
                packs.append({"path": str(tar), "size_mb": round(tar.stat().st_size / 1e6, 2)})
        except Exception:  # noqa: BLE001
            continue
    out["delivery_packs"] = packs
    return out


def judge(art: dict, require_doctor: bool = False) -> dict[str, Any]:
    """按三档判定，返回 {verdict, reasons[], actions[]}。"""
    reasons: list[str] = []
    actions: list[str] = []
    verdict = "ready"

    clean = art.get("clean")
    doctor = art.get("doctor")
    verify = art.get("verify")

    if not clean:
        verdict = "blocked"
        reasons.append("没跑过 03 清洗质检（无 _products/clean/summary.json），没有质检证据")
        actions.append("先跑：python3 run.py clean <编号>")
    else:
        n_rev = int(clean.get("n_review") or 0)
        n_ex = int(clean.get("n_exclude") or 0)
        if n_ex:
            reasons.append(f"03 排除 {n_ex} 集（05 合并会剔除）：{clean.get('excluded_episodes')}")
        if n_rev:
            verdict = "review"
            reasons.append(f"03 有 {n_rev} 集需复核（未剔除）：{clean.get('review_episodes')}")
            actions.append("到 Web 盲审页看 review 集，确认是真实动作就该留")
        dl = (clean.get("dataset_level") or {}).get("signals_summary") or {}
        if dl:
            for tag in ("state", "action"):
                consts = dl.get(f"{tag}_const_dims")
                if consts:
                    verdict = "blocked" if tag == "state" else max(
                        [verdict, "review"], key=lambda v: LEVEL[v])
                    reasons.append(f"{tag} 常量维（通道没数据）：{consts}")
                    actions.append(f"查 {tag} 这些维度是不是没记/没接上，别喂给训练")

    if doctor:
        sev = (doctor.get("overall_severity") or "").upper()
        s = doctor.get("summary") or {}
        reasons.append(f"11 lerobot-doctor: {sev or 'UNKNOWN'} "
                       f"(PASS {s.get('PASS', 0)} / WARN {s.get('WARN', 0)} / FAIL {s.get('FAIL', 0)})")
        if sev == "FAIL":
            verdict = "blocked"
            for c in doctor.get("checks") or []:
                if (c.get("severity") or "").upper() == "FAIL":
                    for m in c.get("messages") or []:
                        reasons.append(f"doctor FAIL · {c.get('name')}: {m.get('message')}")
        elif sev == "WARN":
            verdict = max([verdict, "review"], key=lambda v: LEVEL[v])
    elif require_doctor:
        verdict = max([verdict, "review"], key=lambda v: LEVEL[v])
        reasons.append("未跑 11 第二意见（lerobot-doctor），无法交叉验证")
        actions.append("装好后跑：python3 run.py doctor <编号>")

    if verify is not None:
        if not verify.get("ok", True):
            verdict = "blocked"
            reasons.append("07 结构校验未通过")
            actions.append("看 _products/verify/verify_report.md 修数据后再交付")
        else:
            reasons.append("07 结构校验通过")
    if not art.get("delivery_packs"):
        actions.append("交付时跑：python3 run.py pack <编号>（会带上本 QA 报告）")
    else:
        reasons.append(f"已有交付包：{[p['path'] for p in art['delivery_packs']]}")

    return {"verdict": verdict, "reasons": reasons, "actions": actions}


def build_md(art: dict, verdict: dict) -> list[str]:
    ds_name = art["dataset"]
    icon = {"ready": "✅ ready", "review": "🔍 review", "blocked": "⛔ blocked"}[verdict["verdict"]]
    md = [f"# QA 汇总: {ds_name}", "", f"**结论: {icon}**", ""]
    md += ["## 结论依据", ""]
    md += [f"- {r}" for r in verdict["reasons"]]
    if verdict["actions"]:
        md += ["", "## 建议动作", ""] + [f"- {a}" for a in verdict["actions"]]
    clean = art.get("clean") or {}
    if clean:
        md += ["", "## 质检（03）", "",
               f"- 集数 {clean.get('n_episodes')}：keep {clean.get('n_keep')} / "
               f"review {clean.get('n_review', 0)} / exclude {clean.get('n_exclude')}",
               f"- 逐集明细: `{clean.get('disposition_csv')}`"]
        ds_lv = clean.get("dataset_level") or {}
        agg = ds_lv.get("signals_summary")
        if agg:
            md += ["", "| 统计信号 | 值 |", "|---|---|"]
            for k, v in agg.items():
                md.append(f"| {k} | {' / '.join(f'{a}:{b}' for a, b in v.items()) if isinstance(v, dict) else v} |")
        jo = clean.get("joint_observation")
        if jo:
            md += ["", f"- 关节观测量（只测量）: max|角| {jo.get('max_abs_rad')} rad / "
                       f"max 跳变 {jo.get('max_jump_rad')} rad / 最长零方差 {jo.get('longest_stuck_s')} s"]
    doctor = art.get("doctor")
    if doctor:
        md += ["", "## 第二意见（11 lerobot-doctor）", "",
               f"- 判定: **{doctor.get('overall_severity')}** "
               f"(PASS {(doctor.get('summary') or {}).get('PASS', 0)} / "
               f"WARN {(doctor.get('summary') or {}).get('WARN', 0)} / "
               f"FAIL {(doctor.get('summary') or {}).get('FAIL', 0)})"]
        for c in doctor.get("checks") or []:
            if (c.get("severity") or "").upper() in ("FAIL", "WARN"):
                msgs = [m.get("message") for m in (c.get("messages") or [])][:3]
                md.append(f"- [{c.get('severity')}] {c.get('name')}: {'; '.join(str(m) for m in msgs)}")
    if not doctor:
        md += ["", "## 第二意见（11）", "", "- 未执行（未装 lerobot-doctor 或未跑）"]
    md += ["", "## 产物索引", ""]
    _ds = Path(art["path"])
    for k, v in (("03 质检", str(dataset_io.stage_root(_ds, "clean"))),
                 ("11 第二意见", str(dataset_io.stage_root(_ds, "doctor"))),
                 ("07 校验", str(dataset_io.stage_root(_ds, "verify"))),
                 ("02 时间戳", str(art.get("timestamps_report") or "未跑"))):
        md.append(f"- {k}: `{v}`")
    if art.get("delivery_packs"):
        md.append(f"- 交付包: {[p['path'] for p in art['delivery_packs']]}")
    md += ["", "> 本报告只汇总证据，不改数据：排除判定始终由 03 的 episode_disposition.csv 决定。"]

    # 首行是结论摘要，方便 08/07 打印与台账引用
    return md


def main() -> int:
    ap = argparse.ArgumentParser(description="12 汇总 QA 证据（只读）")
    ap.add_argument("--input", required=True, help="数据集目录")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", help="输出目录（默认 <ds>_products/qa）")
    ap.add_argument("--require-doctor", action="store_true", help="没跑 11 就判 review")
    ap.add_argument("--fail-on", choices=["ready", "review", "blocked"], default=None,
                    help="达到该档及以上时返回非 0（便于 CI/脚本门禁）")
    args = ap.parse_args()

    ds = Path(args.input).expanduser()
    kind, reason = dataset_io.detect_dataset(ds)
    if kind not in ("v2.1", "v3.0"):
        print(f"[ERROR] 不是数据集: {ds}（{reason}）")
        return 1
    cfg: dict = {}
    if Path(args.config).is_file():
        try:
            import yaml
            cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001
            cfg = {}

    art = collect(ds, cfg)
    verdict = judge(art, require_doctor=args.require_doctor)
    out_root = Path(args.out) if args.out else dataset_io.new_stage_dir(ds, "qa")
    out_root.mkdir(parents=True, exist_ok=True)
    payload = {"dataset": ds.name, "path": str(ds), "generated_at":
               __import__("datetime").datetime.now().isoformat(timespec="seconds"),
               "verdict": verdict["verdict"], "reasons": verdict["reasons"],
               "actions": verdict["actions"],
               "clean_summary": art.get("clean"), "doctor": art.get("doctor"),
               "verify": art.get("verify"), "delivery_packs": art.get("delivery_packs")}
    report.write_json(out_root / "qa_summary.json", payload)
    report.write_md(out_root / "qa_summary.md", build_md(art, verdict))

    icon = {"ready": "✅", "review": "🔍", "blocked": "⛔"}[verdict["verdict"]]
    print(f"[OK] {ds.name}: QA {icon} {verdict['verdict']} -> {out_root / 'qa_summary.md'}")
    for r in verdict["reasons"]:
        print(f"       · {r}")
    for a in verdict["actions"]:
        print(f"       → {a}")
    if args.fail_on and LEVEL[verdict["verdict"]] >= LEVEL[args.fail_on]:
        print(f"[!] QA 结论 {verdict['verdict']} 达到 --fail-on={args.fail_on}，返回非 0")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
