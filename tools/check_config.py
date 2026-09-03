#!/usr/bin/env python3
"""配置检查：YAML 解析 + 结构/类型/取值校验，秒级反馈（install 后可先跑它）。

用法:
    python3 tools/check_config.py               # 检查 ./config.yaml
    python3 tools/check_config.py --file 别的.yaml
退出码：0=全部通过；1=有错误（脚本可直接拿来当"装完先体检"）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

QC_KEYS = {
    "min_duration_s": float,
    "max_duration_s": float,
    "fps_deviation": float,
    "max_drop_ratio": float,
    "joint_limits_rad": float,
    "joint_jump_rad": float,
    "stuck_s": float,
    "blur_laplacian_thr": float,
    "blur_bad_ratio": float,
    "blur_sample_frames": int,
}

VER = tuple(sys.version_info[:2])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="config.yaml")
    args = ap.parse_args()
    p = Path(args.file)
    ok = True
    errs, warns = [], []

    if not p.is_file():
        print(f"[ERR] 配置文件不存在: {p}（先 cp config.example.yaml config.yaml）")
        return 1

    try:
        import yaml
        with open(p, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as e:  # noqa: BLE001
        print(f"[ERR] YAML 解析失败: {e}")
        return 1

    if not isinstance(cfg, dict):
        print("[ERR] 配置顶层应为映射（字典）")
        return 1

    # --- paths
    paths = cfg.get("paths")
    if not isinstance(paths, dict):
        errs.append("paths 缺失或不是映射（需要 batches/output/ledger 三个路径）")
    else:
        for k in ("batches", "output", "ledger"):
            v = paths.get(k)
            if not isinstance(v, str) or not v.strip():
                errs.append(f"paths.{k} 缺失或非字符串")
            elif not Path(v).is_absolute():
                warns.append(f"paths.{k} 不是绝对路径: {v}（建议写 /xxx 绝对路径）")

    # --- robot_type_map
    rtm = cfg.get("robot_type_map")
    if rtm is not None and not isinstance(rtm, dict):
        errs.append("robot_type_map 应为映射（机型编号 -> 简称），可留空")

    # --- defaults
    dft = cfg.get("defaults")
    if dft is not None:
        if not isinstance(dft, dict):
            errs.append("defaults 应为映射")
        else:
            if dft.get("fps_nominal") is not None and not isinstance(dft.get("fps_nominal"), (int, float)):
                errs.append("defaults.fps_nominal 应为数字")
            if dft.get("version") is not None and not str(dft.get("version")).startswith("v"):
                warns.append(f"defaults.version 建议以 v 开头，当前: {dft.get('version')}")

    # --- qc
    qc = cfg.get("qc")
    if qc is not None:
        if not isinstance(qc, dict):
            errs.append("qc 应为映射")
        else:
            for k, v in qc.items():
                if k not in QC_KEYS:
                    warns.append(f"qc.{k} 不在已知列表（{sorted(QC_KEYS)}），可能是笔误或被忽略的旧字段")
                elif not isinstance(v, (int, float)) or isinstance(v, bool):
                    errs.append(f"qc.{k} 应为数字，当前 {type(v).__name__}")
                elif k == "blur_sample_frames" and not isinstance(v, int):
                    errs.append(f"qc.blur_sample_frames 应为整数，当前 {v}")

    # --- annotate / merge（预留）
    for sec in ("annotate", "merge"):
        v = cfg.get(sec)
        if v is not None and not isinstance(v, dict):
            errs.append(f"{sec} 应为映射（预留段，可留空）")

    print(f"配置文件: {p}")
    for e in errs:
        print(f"  [ERR]  {e}")
        ok = False
    for w in warns:
        print(f"  [WARN] {w}")
    if not errs:
        print(f"  [OK] 结构合法（{len(warns)} 条提示）")
    if errs:
        print(f"\n结果: 失败（{len(errs)} 处错误）——请修正后重试")
    else:
        print(f"\n结果: 通过（{len(warns)} 条建议可不改）")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())