#!/usr/bin/env python3
"""05 合并：把若干 v2.1 批次（由你显式指定）合并成一个 v2.1 数据集。

内核移植自采集团队已验证的 merge_lerobot_v21_arx_bimanual.py：
  - 保持 v2.1 布局，episode 从 0 重编号，index 全局连续，task_index 重映射；
  - 视频直拷（不转码），meta(info/tasks/episodes/episodes_stats/splits) 全套重写；
  - 安全校验：≥2 个源、codebase_version=v2.1、fps/robot_type/features/
    chunks_size/data_path/video_path 必须一致。

本步新增：
  - 坏集排除：每个输入自动查找 <输入>_clean/episode_disposition.csv，
    verdict=exclude 的整集跳过（parquet/视频/meta 都不进输出）；
    可用 --dispositions 显式指定，或用 --no-exclude 强制全并。
  - 输出命名：默认按命名公式 {task}_{robot}_{MMDD[-MMDD]}_{N}cam_v{ver}
    从源目录/数据/config 自动生成；也可 --output 手动指定。
  - 防呆：全部被排除 -> 报错不产出；parquet 长度与 meta 不一致 -> 报错。

用法:
    python3 pipe/05_merge.py --inputs 目录A 目录B [目录C...] [--output 输出]
          [--dispositions csv...] [--no-exclude] [--overwrite] [--config config.yaml]
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipe.lib import dataset_io  # noqa: E402


# ---------------------------------------------------------------- 基础 IO
def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def comparable_features(info: dict[str, Any]) -> dict[str, Any]:
    return info["features"]


# ---------------------------------------------------------------- 校验
def validate_sources(sources: list[Path]) -> dict[str, Any]:
    if len(sources) < 2:
        raise ValueError("至少需要 2 个源数据集（想合并哪些就勾哪些，但合并得 ≥2 个）")
    base_info = None
    for root in sources:
        kind, reason = dataset_io.detect_dataset(root)
        if kind != "v2.1":
            raise ValueError(f"{root}: 不是 v2.1 数据集（{reason}）")
        info = dataset_io.read_meta(root)["info"]
        ver = str(info.get("codebase_version") or "")
        if ver not in ("v2.1", "2.1"):
            raise ValueError(f"{root}: codebase_version={ver or '<空>'}，期望 v2.1")
        if base_info is None:
            base_info = info
            continue
        for key in ("fps", "robot_type", "data_path", "video_path", "chunks_size"):
            if info.get(key) != base_info.get(key):
                raise ValueError(f"{root}: info['{key}'] 与第一个源不一致 (不可混机型/帧率合并)")
        if comparable_features(info) != comparable_features(base_info):
            raise ValueError(f"{root}: features 与第一个源不一致（相机位置/分辨率需相同）")
    return base_info


# ---------------------------------------------------------------- 坏集排除
def load_disposition(csv_path: Path | None) -> dict[int, str] | None:
    """读 03 的 episode_disposition.csv -> {episode: verdict}；文件不存在返回 None。"""
    if csv_path is None or not csv_path.is_file():
        return None
    out: dict[int, str] = {}
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            try:
                out[int(row["episode"])] = str(row.get("verdict") or "keep").strip()
            except (ValueError, KeyError):
                continue
    return out


def discover_dispositions(inputs: list[Path], explicit: list[str | None],
                          no_exclude: bool) -> list[dict[int, str] | None]:
    """每个源 -> 处置表（None=无清单，全并入）。显式 --dispositions 优先。"""
    if no_exclude:
        return [None] * len(inputs)
    out: list[dict[int, str] | None] = []
    for i, src in enumerate(inputs):
        if explicit and i < len(explicit) and explicit[i]:
            cand = Path(explicit[i])
        else:
            cand = src.parent / f"{src.name}_clean" / "episode_disposition.csv"
        disp = load_disposition(cand)
        if disp is None and not (explicit and i < len(explicit) and explicit[i]):
            print(f"[WARN] 未找到 {src.name} 的处置清单（{cand}），该源将全部并入；"
                  f"建议先跑 03_clean，或用 --no-exclude 确认不排除")
        out.append(disp)
    return out


# ---------------------------------------------------------------- 输出命名
def auto_output_name(inputs: list[Path], cfg: dict) -> str:
    base_info = dataset_io.read_meta(inputs[0])["info"]
    defaults = cfg.get("defaults", {})
    robot_map = cfg.get("robot_type_map", {})

    robot = robot_map.get(str(base_info.get("robot_type")), "unk")
    if robot == "unk":
        print(f"[!] robot_type '{base_info.get('robot_type')}' 未在 robot_type_map，输出名用 unk")

    # 日期：取所有源的 MM/DD 范围
    starts, ends = [], []
    for src in inputs:
        pr = dataset_io.parse_date_range(src.name)
        if pr:
            starts.append(pr[0])
            ends.append(pr[1])
    if starts:
        lo, hi = min(starts), max(ends)
        date_str = f"{lo}-{hi}" if lo != hi else lo
    else:
        print("[WARN] 无法从目录名解析日期，输出名日期部分用 ???")
        date_str = "????-????"

    # 任务：config.defaults.task > 源内单任务 > 多任务提示用 merged
    task = (defaults.get("task") or "").strip()
    if not task:
        tasks = dataset_io.read_meta(inputs[0])["tasks"]
        names = {t.get("task") for t in tasks if t.get("task")}
        if len(names) == 1:
            task = next(iter(names))
        elif len(names) > 1:
            print("[!] 源含多任务且未配 defaults.task，输出名任务部分用 merged")
            task = "merged"
        else:
            task = "task"
    task = task.replace(" ", "_")

    ncam = len(video_keys_of(base_info))
    version = defaults.get("version", "v1")
    return f"{task}_{robot}_{date_str}_{ncam}cam_{version}"


def video_keys_of(info: dict[str, Any]) -> list[str]:
    """相机键：先按真实 v2.1 的 features[dtype=video]；兜底 info['videos'] 的键。"""
    keys = [str(k) for k, v in (info.get("features") or {}).items()
            if isinstance(v, dict) and v.get("dtype") == "video"]
    if not keys:
        keys = [str(k) for k, v in (info.get("videos") or {}).items() if isinstance(v, dict)]
    return keys


# ---------------------------------------------------------------- 合并主流程
def merge_datasets(sources: list[Path], output: Path, dispositions: list[dict[int, str] | None],
                   overwrite: bool = False) -> dict[str, int]:
    base_info = validate_sources(sources)
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"{output} 已存在。加 --overwrite 才会覆盖，或换个 --output")
        shutil.rmtree(output)

    chunks_size = int(base_info.get("chunks_size", 1000))
    video_keys = video_keys_of(base_info)
    output.mkdir(parents=True)
    (output / "meta").mkdir(parents=True, exist_ok=True)

    # 任务表合并（跨源去重）
    task_name_to_idx: dict[str, int] = {}
    all_task_rows: list[dict[str, Any]] = []
    source_task_maps: list[dict[int, int]] = []
    for src in sources:
        rows = dataset_io.read_meta(src)["tasks"]
        old_to_new: dict[int, int] = {}
        for row in rows:
            task = row.get("task")
            if task is None:
                continue
            if task not in task_name_to_idx:
                task_name_to_idx[task] = len(task_name_to_idx)
                all_task_rows.append({"task_index": task_name_to_idx[task], "task": task})
            old_to_new[int(row.get("task_index", 0))] = task_name_to_idx[task]
        source_task_maps.append(old_to_new)

    # episodes_stats：官方 v2.1→v3.0 转换器硬性需要；源缺时从 parquet 补算
    source_stats = []
    n_missing_stats = 0
    for src in sources:
        p = src / "meta/episodes_stats.jsonl"
        if p.is_file():
            source_stats.append({int(r["episode_index"]): r for r in read_jsonl(p)})
        else:
            source_stats.append({})
            n_missing_stats += 1
    if n_missing_stats:
        print(f"[i] 有 {n_missing_stats} 个源缺 episodes_stats.jsonl，将从 parquet 补算（官方转换器需要）")

    merged_episodes: list[dict[str, Any]] = []
    merged_stats: list[dict[str, Any]] = []
    total_frames = 0
    total_episodes = 0
    per_source: list[tuple[str, int, int]] = []

    for src_idx, src in enumerate(sources):
        info = dataset_io.read_meta(src)["info"]
        episodes = {int(r["episode_index"]): r for r in read_jsonl(src / "meta/episodes.jsonl")}
        stats = source_stats[src_idx]
        task_map = source_task_maps[src_idx]
        disp = dispositions[src_idx]
        src_chunks = int(info.get("chunks_size", chunks_size))

        n_in, n_out = 0, 0
        for old_ep in sorted(episodes):
            if disp and disp.get(old_ep) == "exclude":
                n_out += 1
                continue
            ep_row = dict(episodes[old_ep])
            new_ep = total_episodes
            new_chunk = new_ep // chunks_size
            old_chunk = old_ep // src_chunks

            src_data = src / f"data/chunk-{old_chunk:03d}/episode_{old_ep:06d}.parquet"
            dst_data = output / f"data/chunk-{new_chunk:03d}/episode_{new_ep:06d}.parquet"
            if not src_data.is_file():
                raise FileNotFoundError(f"{src_data}")
            df = pd.read_parquet(src_data)
            length = int(len(df))
            if length != int(ep_row["length"]):
                raise ValueError(f"{src_data}: parquet 长度 {length} != meta 长度 {ep_row['length']}")

            df = df.copy()
            df["episode_index"] = new_ep
            df["index"] = range(total_frames, total_frames + length)
            if "task_index" in df.columns:
                df["task_index"] = df["task_index"].map(lambda x: task_map[int(x)])
            dst_data.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(dst_data, index=False)

            for vk in video_keys:
                src_v = src / f"videos/chunk-{old_chunk:03d}/{vk}/episode_{old_ep:06d}.mp4"
                dst_v = output / f"videos/chunk-{new_chunk:03d}/{vk}/episode_{new_ep:06d}.mp4"
                if not src_v.is_file():
                    raise FileNotFoundError(f"{src_v}")
                dst_v.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_v, dst_v)

            ep_row["episode_index"] = new_ep
            ep_row["tasks"] = sorted(set(ep_row.get("tasks", [])))
            ep_row["length"] = length
            merged_episodes.append(ep_row)

            s = stats.get(old_ep)
            if s is None:
                s = dataset_io.compute_episode_stats(df, new_ep)
            else:
                s = dict(s)
                s["episode_index"] = new_ep
            merged_stats.append(s)

            total_frames += length
            total_episodes += 1
            n_in += 1
        per_source.append((src.name, n_in, n_out))
        print(f"  [{src.name}] 并入 {n_in} 集 / 排除 {n_out} 集")

    if total_episodes == 0:
        raise ValueError("所有源的所有 episode 均被排除，拒绝产出空数据集（请检查处置清单或 --no-exclude）")

    merged_info = dict(base_info)
    merged_info["total_episodes"] = total_episodes
    merged_info["total_frames"] = total_frames
    merged_info["total_tasks"] = len(all_task_rows)
    merged_info["total_chunks"] = (total_episodes - 1) // chunks_size + 1
    merged_info["total_videos"] = len(video_keys)
    merged_info["splits"] = {"train": f"0:{total_episodes}"}

    write_json(output / "meta/info.json", merged_info)
    write_jsonl(output / "meta/tasks.jsonl", all_task_rows)
    write_jsonl(output / "meta/episodes.jsonl", merged_episodes)
    write_jsonl(output / "meta/episodes_stats.jsonl", merged_stats)

    print(f"[OK] 合并完成 -> {output}")
    print(f"[OK] 总集数 {total_episodes} / 总帧数 {total_frames} / 任务数 {len(all_task_rows)}")
    return {"episodes": total_episodes, "frames": total_frames}


def main() -> int:
    ap = argparse.ArgumentParser(description="05 合并：显式指定 2+ 个 v2.1 批次合并成一集")
    ap.add_argument("--inputs", nargs="+", required=True, help="要合并的数据集目录（你指定，2 个或更多）")
    ap.add_argument("--output", default=None, help="输出目录（默认自动命名）")
    ap.add_argument("--dispositions", nargs="+", default=None,
                    help="各源处置 csv（与 --inputs 一一对应；空串=无清单；缺省自动找 <输入>_clean/）")
    ap.add_argument("--no-exclude", action="store_true", help="不排除任何集（忽略处置清单）")
    ap.add_argument("--overwrite", action="store_true", help="输出目录已存在时覆盖")
    ap.add_argument("--config", default="config.yaml", help="配置文件（命名用，可不存在）")
    args = ap.parse_args()

    inputs = [Path(x).expanduser().resolve() for x in args.inputs]
    cfg = {}
    if Path(args.config).is_file():
        import yaml
        with open(args.config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    explicit_disp: list[str | None] = list(args.dispositions or [])
    explicit_disp += [None] * (len(inputs) - len(explicit_disp))
    dispositions = discover_dispositions(inputs, explicit_disp, args.no_exclude)

    if args.output:
        output = Path(args.output).expanduser().resolve()
    else:
        out_root = cfg.get("paths", {}).get("output") or inputs[0].parent
        output = Path(out_root).expanduser().resolve() / auto_output_name(inputs, cfg)
        print(f"[i] 自动输出：{output.name}")

    try:
        stats = merge_datasets(inputs, output, dispositions, overwrite=args.overwrite)
    except (ValueError, FileExistsError, FileNotFoundError) as e:
        print(f"[ERROR] {e}")
        return 1
    print(f"[OK] 结果目录: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())