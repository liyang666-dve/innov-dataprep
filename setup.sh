#!/usr/bin/env bash
# innov-dataprep 一次性配置：探测环境 + 装依赖。克隆后跑一次即可。
set -euo pipefail
cd "$(dirname "$0")"

PYTHON_BIN=""
echo "==> 探测 Python 环境（优先复用已有 lerobot conda 环境）..."
if command -v conda >/dev/null 2>&1; then
  for env in "${LEROBOT_ENV:-lerobot_arx_sdk311}" lerobot; do
    if conda env list 2>/dev/null | grep -qE "^\s*${env}\s"; then
      PYTHON_BIN="$(conda run -n "$env" which python 2>/dev/null | tail -1)"
      [ -n "$PYTHON_BIN" ] && echo "    复用 conda 环境: ${env} -> ${PYTHON_BIN}" && break
    fi
  done
fi
if [ -z "$PYTHON_BIN" ]; then
  PYTHON_BIN="$(command -v python3 || true)"
  echo "    使用系统 python3: $PYTHON_BIN"
fi

echo "==> 安装/校验自研部分依赖（numpy pandas pyarrow PyYAML）..."
"$PYTHON_BIN" -m pip install --quiet --upgrade numpy pandas pyarrow PyYAML

echo "==> 检查 ffprobe（视频帧数核对用，缺少则该项自动跳过）..."
if command -v ffprobe >/dev/null 2>&1; then echo "    ffprobe ✓"; else echo "    ⚠ 未检测到 ffprobe（提示：sudo apt install ffmpeg）"; fi

echo ""
echo "==> 完成。下一步："
echo "    cp config.example.yaml config.yaml   # 填路径/机器人映射/默认值"
echo "    然后: python3 pipe/01_inspect.py --help"