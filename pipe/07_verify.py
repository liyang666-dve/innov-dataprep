#!/usr/bin/env python3
"""07 校验：交付前结构 smoke + 交付 sha256 清单。

跑什么（v2.1 / v3.0 通用，不依赖 lerobot）：
  1. 识别格式（v2.1 / v3.0 / 不是数据集）
  2. meta/info.json 字段（fps/robot_type/features）
  3. 任务表可读（v2.1 tasks.jsonl / v3.0 tasks.parquet）
  4. 集数一致：meta 记录数 vs 实际数据文件
  5. 帧数一致：parquet 实际行数总数 vs meta 标称总行数（pyarrow 页脚，不整读）
  6. 视频存在性：v2.1 每集每机位有 mp4；v3.0 每机位有 chunk 文件
  7. 数据 chunk 编号连续（0 起）
  8. （v2.1）episodes_stats.jsonl 存在性提示（06 转换需要，缺了 06 会补算）
  9. （v3.0）episodes 元数据含 length/视频时间戳区间

产物：<名>_products/verify/verify_report.json + verify_report.md（默认）。

另有 --delivery <tar.gz>：整包校验（解压 → 逐个文件 sha256 比对清单 → 结构 smoke），
供训练机收到交付包后自查：python3 pipe/07_verify.py --delivery xxx_delivery.tar.gz

用法:
    python3 pipe/07_verify.py --input <数据集> [--input ...] [--out 目录]
    python3 pipe/07_verify.py --delivery <交付包.tar.gz>
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipe.lib import dataset_io, report  # noqa: E402


# ---------------------------------------------------------------- sha256
def sha256_of_file(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def build_sha256_manifest(ds: Path) -> list[tuple[str, str]]:
    """返回 [(相对路径, sha256)]，按相对路径排序。相对路径用 / 分隔（打包后一致）。"""
    rows = []
    for p in sorted(ds.rglob("*")):
        if p.is_file():
            rel = p.relative_to(ds).as_posix()
            rows.append((rel, sha256_of_file(p)))
    return rows


# ---------------------------------------------------------------- 结构检查
def _parquet_rows(path: Path) -> int:
    import pyarrow.parquet as pq
    try:
        return int(pq.read_metadata(path).num_rows)
    except Exception:  # noqa: BLE001
        return -1


def _v3_episode_rows(eps_meta) -> int:
    try:
        return int(eps_meta["length"].sum()) if eps_meta is not None and len(eps_meta) else 0
    except Exception:  # noqa: BLE001
        return -1


def verify_dataset(ds: Path) -> dict:
    """对单个数据集做结构 smoke，返回 report dict。"""
    items: list[dict] = []
    warn = lambda m: items.append({"status": "WARN", "detail": m})  # noqa: E731
    fail = lambda m: items.append({"status": "FAIL", "detail": m})  # noqa: E731
    ok = lambda m: items.append({"status": "OK", "detail": m})      # noqa: E731

    kind, reason = dataset_io.detect_dataset(ds)
    if kind not in ("v2.1", "v3.0"):
        return {"name": ds.name, "format": kind, "items": [{"status": "FAIL", "detail": f"不是数据集: {reason}"}], "ok": False}

    meta = dataset_io.read_meta(ds)
    info = meta["info"]
    fps = info.get("fps")
    if isinstance(fps, (int, float)) and fps > 0:
        ok(f"info.fps = {fps}")
    else:
        fail(f"info.fps 缺失或非法: {fps!r}")
    rt = info.get("robot_type")
    if rt:
        ok(f"robot_type = {rt}")
    else:
        warn("robot_type 为空（登记时记为 unk）")
    feats = info.get("features") or {}
    if isinstance(feats, dict) and feats:
        ok(f"features {len(feats)} 项")
    else:
        warn("features 为空（相机/机位信息缺失，视频核对会退化，不影响其余）")

    # 任务表
    tasks = meta.get("tasks") or []
    if tasks:
        ok(f"任务表 {len(tasks)} 条")
    else:
        warn("任务表为空/不可读")

    # 集数与帧数
    eps_data = dataset_io.discover_episodes(ds) if kind == "v2.1" else []
    if kind == "v2.1":
        eps_meta_rows = len(meta.get("episodes_meta") or [])
        actual_rows = sum(_parquet_rows(p) for p in eps_data)
        meta_rows = sum(int(r.get("length") or 0) for r in (meta.get("episodes_meta") or []))
        n_eps = len(eps_data)
        if eps_meta_rows and eps_meta_rows != n_eps:
            fail(f"集数不一致: meta {eps_meta_rows} 条 vs 数据文件 {n_eps} 个")
        elif n_eps:
            ok(f"集数 {n_eps}")
        else:
            fail("没有数据文件")
        if meta_rows and actual_rows != meta_rows:
            fail(f"帧数不一致: parquet 共 {actual_rows} 行 vs meta 标称 {meta_rows}")
        else:
            ok(f"帧数合计 {actual_rows}")
        stats_p = ds / "meta" / "episodes_stats.jsonl"
        if stats_p.is_file():
            ok("episodes_stats.jsonl 存在（06 转换可直接用）")
        else:
            warn("缺 episodes_stats.jsonl（06 转换时会自动补算，不影响）")
        total_frames = actual_rows
    else:
        eps_meta_pd = dataset_io._v3_episodes_meta(ds)
        n_eps = _v3_n_eps(eps_meta_pd)
        data_files = sorted((ds / "data").glob("**/file-*.parquet")) or sorted((ds / "data").glob("**/*.parquet"))
        actual_rows = sum(_parquet_rows(p) for p in data_files)
        meta_rows = _v3_episode_rows(eps_meta_pd)
        if n_eps:
            ok(f"集数 {n_eps}")
        else:
            fail("meta/episodes 读不到集数")
        if meta_rows and actual_rows != meta_rows:
            fail(f"帧数不一致: data 共 {actual_rows} 行 vs episodes meta 标称 {meta_rows}")
        else:
            ok(f"帧数合计 {actual_rows}")
        if "length" in (eps_meta_pd.columns if eps_meta_pd is not None else []):
            ok("episodes meta 含 length")
        else:
            warn("episodes meta 缺 length 列")
        vcols = [c for c in (eps_meta_pd.columns if eps_meta_pd is not None else []) if "from_timestamp" in c]
        if vcols:
            ok(f"episodes meta 含视频时间戳区间 {len(vcols)} 列")
        else:
            warn("episodes meta 缺视频 from/to_timestamp（视频帧数核对将不可用）")
        total_frames = actual_rows

    # chunk 编号连续
    chunk_nums = sorted({int(p.name.split("-")[1]) for p in (ds / "data").glob("chunk-*")
                         if p.name.startswith("chunk-") and p.name.split("-")[1].isdigit()})
    if chunk_nums == list(range(max(chunk_nums, default=-1) + 1)):
        ok(f"data chunk 连续: 0-{chunk_nums[-1] if chunk_nums else '?'}")
    else:
        warn(f"data chunk 编号不连续: {chunk_nums}")

    # 视频存在性
    cams = dataset_io.camera_layout(ds, info)
    if kind == "v2.1":
        missing = 0
        for p in eps_data:
            ep = dataset_io.episode_index(p)
            chunk = p.parent.name
            for cam in cams:
                if not (ds / "videos" / chunk / cam / f"episode_{ep:06d}.mp4").is_file():
                    missing += 1
        if missing:
            fail(f"缺 {missing} 个视频文件（逐集逐机位核对）")
        else:
            ok(f"视频齐全（{len(cams)} 机位 × {len(eps_data)} 集）" if eps_data else "无数据可核")
    else:
        vmiss = []
        for cam in cams:
            files = sorted((ds / "videos" / cam).glob("**/file-*.mp4"))
            if not files:
                vmiss.append(cam)
        if vmiss:
            warn(f"v3.0 视频缺失机位: {vmiss}")
        else:
            ok(f"v3.0 视频存在（{len(cams)} 机位）" if cams else "无相机定义")

    n_warn = sum(1 for i in items if i["status"] == "WARN")
    n_fail = sum(1 for i in items if i["status"] == "FAIL")
    return {
        "name": ds.name,
        "format": kind,
        "n_episodes": n_eps,
        "total_frames": total_frames,
        "cameras": list(cams),
        "items": items,
        "n_warn": n_warn,
        "n_fail": n_fail,
        "ok": n_fail == 0,
    }


def _v3_n_eps(eps_meta) -> int:
    if eps_meta is None or len(eps_meta) == 0:
        return 0
    try:
        return int(eps_meta["episode_index"].nunique())
    except Exception:  # noqa: BLE001
        return int(len(eps_meta))


def fmt_report(ds_report: dict) -> list[str]:
    verdict = "PASS" if ds_report["ok"] else "FAIL"
    lines = [
        f"# 校验报告: {ds_report['name']}",
        "",
        f"- 格式: {ds_report['format']} / 集数: {ds_report['n_episodes']} / 帧数: {ds_report['total_frames']} / 相机: {', '.join(ds_report['cameras']) or '—'}",
        f"- 结论: **{verdict}**（{ds_report['n_fail']} 失败 / {ds_report['n_warn']} 警告）",
        "",
        "| 状态 | 检查项 |",
        "|---|---|",
    ]
    for i in ds_report["items"]:
        lines.append(f"| {i['status']} | {i['detail']} |")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description="07 校验：交付前结构 smoke + 交付 sha256 清单")
    ap.add_argument("--input", action="append", help="数据集目录（v2.1/v3.0），可多次")
    ap.add_argument("--delivery", help="交付包 .tar.gz：解压→sha256 逐文件比对→结构 smoke（训练机用）")
    ap.add_argument("--out", help="报告输出目录（默认 <名>_products/verify）")
    ap.add_argument("--no-sha", action="store_true", help="不生成数据集 sha256 清单")
    args = ap.parse_args()

    if args.delivery:
        pkg = Path(args.delivery).expanduser()
        if not pkg.is_file():
            print(f"[ERROR] 交付包不存在: {pkg}")
            return 1
        return verify_delivery(pkg, args)
    if not args.input:
        print("[ERROR] 需要 --input <数据集> 或 --delivery <交付包>")
        return 2

    rc = 0
    for s in args.input:
        ds = Path(s).expanduser()
        r = verify_dataset(ds)
        out = Path(args.out) if args.out else dataset_io.new_stage_dir(ds, "verify")
        report.write_json(out / "verify_report.json", r)
        report.write_md(out / "verify_report.md", fmt_report(r))
        if not args.no_sha and r["ok"]:
            rows = build_sha256_manifest(ds)
            (out / "dataset_sha256sums.txt").write_text(
                "\n".join(f"{h}  {rel}" for rel, h in rows) + "\n", encoding="utf-8")
            print(f"[OK] {ds.name}: 数据集 sha256 清单 {len(rows)} 个文件 -> {out / 'dataset_sha256sums.txt'}")
        if r["ok"]:
            print(f"[OK] 校验通过: {ds.name}（{r['format']}, {r['n_episodes']} 集, {r['total_frames']} 帧, {r['n_warn']} 警告）")
        else:
            print(f"[FAIL] 校验不通过: {ds.name}（{r['n_fail']} 项失败）-> {out / 'verify_report.md'}")
            rc = 1
    return rc


def verify_delivery(pkg: Path, args: argparse.Namespace) -> int:
    """交付包整包校验：解压到临时目录 → sha256 逐文件比对 → 结构 smoke。"""
    tmp = Path(tempfile.mkdtemp(prefix="innov_verify_"))
    try:
        with tarfile.open(pkg, "r:*") as tf:
            names = tf.getnames()
            for m in names:
                # 压缩包内路径防穿越
                if m.startswith("/") or ".." in m.split("/"):
                    print(f"[ERROR] 交付包含非法路径: {m}")
                    return 1
            tf.extractall(tmp)
        members = [tmp / m for m in names if not m.endswith("/")]
        manifest = [m for m in members if m.name == "sha256sums.txt"]
        # 数据集根：sha256sums.txt 所在目录（打包时清单放在数据集根内）
        ds_root = manifest[0].parent if manifest else tmp
        if not manifest:
            print("[WARN] 交付包内无 sha256sums.txt，跳过逐文件比对")
        else:
            bad = 0
            for line in manifest[0].read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                hexv, _, rel = line.partition("  ")
                f = ds_root / rel
                if not f.is_file() or sha256_of_file(f) != hexv:
                    print(f"[FAIL] sha256 不一致: {rel}")
                    bad += 1
            if bad:
                print(f"[ERROR] 交付包 {bad} 个文件校验失败，包可能损坏")
                return 1
            print(f"[OK] sha256 逐文件比对通过（{len([l for l in manifest[0].read_text(encoding='utf-8').splitlines() if l.strip()])} 个文件）")
        r = verify_dataset(ds_root)
        print("")
        for ln in fmt_report(r):
            print(ln)
        return 0 if r["ok"] else 1
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())