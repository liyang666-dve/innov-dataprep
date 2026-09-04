#!/usr/bin/env python3
"""09 标注：VLM(OpenAI 兼容接口) 逐集生成数据质量评分+处理建议。

只读；未启用/缺端点/缺 Key 时优雅拦截（config 校验），不修改数据。
产物写 <dataset>_annotation/suggestions.jsonl + summary.json。

用法:
    python3 pipe/09_annotate.py --input <dataset> [--input ...] [--config config.yaml]
环境变量: 按 config.yaml annotate.api_key_env 指定，如 export DEEPSEEK_API_KEY=xxx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipe.lib import dataset_io, suggest  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", action="append", required=True, help="v2.1/v3.0 数据集路径（可多次）")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    a = ap.parse_args(argv)
    cfg = suggest.load_yaml(a.config)
    rc = 0
    for s in a.input:
        ds = Path(s)
        kind, reason = dataset_io.detect_dataset(ds)
        if kind not in ("v2.1", "v3.0"):
            print(f"[跳过] {ds.name} 不是 v2.1/v3.0 数据集: {reason}")
            rc = 1
            continue
        try:
            rc = suggest.annotate_batch(ds, None, cfg, emit=print) or rc
        except KeyboardInterrupt:
            print("[!] 已中断")
            return 1
    return rc


if __name__ == "__main__":
    sys.exit(main())