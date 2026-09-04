#!/usr/bin/env bash
# innov-dataprep 一次性配置：探测 Python 环境 + 装依赖。克隆后跑一次即可。
# 策略（按顺序）：conda lerobot 环境 -> 本仓 .venv -> 系统 python3（PEP668 时退 --break-system-packages）。
# 注意：pip 安装一律不加 --upgrade（避免把 lerobot 环境里已装好的 numpy/opencv 升级弄坏）。
set -uo pipefail
cd "$(dirname "$0")"

PYTHON_BIN=""
MODE=""

echo "==> 1/4 探测 Python 环境（优先复用已有 lerobot conda 环境）..."
if command -v conda >/dev/null 2>&1; then
  for env in "${LEROBOT_ENV:-lerobot_arx_sdk311}" lerobot; do
    if conda env list 2>/dev/null | grep -qE "^\s*${env}\s"; then
      PYTHON_BIN="$(conda run -n "$env" which python 2>/dev/null | tail -1)"
      if [ -n "$PYTHON_BIN" ]; then
        MODE="conda:${env}"
        echo "    复用 conda 环境: ${env} -> ${PYTHON_BIN}"
        break
      fi
    fi
  done
fi
if [ -z "$PYTHON_BIN" ] && [ -x ".venv/bin/python" ]; then
  PYTHON_BIN="$(pwd)/.venv/bin/python"
  MODE="venv"
  echo "    复用已有 .venv: $PYTHON_BIN"
fi
if [ -z "$PYTHON_BIN" ]; then
  SYS="$(command -v python3 || true)"
  if [ -n "$SYS" ] && "$SYS" -m venv .venv >/dev/null 2>&1; then
    PYTHON_BIN="$(pwd)/.venv/bin/python"
    MODE="venv(new)"
    echo "    已创建 .venv（避免污染系统环境）: $PYTHON_BIN"
  else
    PYTHON_BIN="$SYS"
    MODE="system"
    echo "    使用系统 python3（无法建 venv，可能遇到 PEP 668 限制，会自动兜底）: $PYTHON_BIN"
  fi
fi
[ -n "$PYTHON_BIN" ] || { echo "[ERROR] 未找到 python3，请先安装 Python 3.10+"; exit 1; }

echo "==> 2/4 安装/校验依赖（core + 视频/模糊/Rerun 回放用，不加 --upgrade）..."
pip_install() {
  "$PYTHON_BIN" -m pip install --quiet "$@" 2>/tmp/innov_pip_err.txt
  local rc=$?
  if [ $rc -ne 0 ] && grep -qi "externally-managed-environment" /tmp/innov_pip_err.txt; then
    echo "    ⚠ 检测到系统 Python 受管理（PEP 668），改用 --break-system-packages 重试..."
    "$PYTHON_BIN" -m pip install --quiet --break-system-packages "$@" 2>/tmp/innov_pip_err2.txt
    rc=$?
  fi
  return $rc
}
if ! pip_install numpy pandas pyarrow PyYAML; then
  echo "[ERROR] 核心依赖安装失败（见上方输出）。conda 用户请检查网络；venv 用户请确认 pip 可用。"
  exit 1
fi
if ! pip_install opencv-python-headless av rerun-sdk; then
  echo "    ⚠ 可选依赖（opencv/av/rerun-sdk）安装失败，模糊检查/视频解码/Rerun 回放不可用，其余功能不受影响。"
fi

echo "==> 3/4 检查 ffprobe（视频帧数核对用，缺少则该项明确提示、自动跳过）..."
if command -v ffprobe >/dev/null 2>&1; then
  echo "    ffprobe ✓"
else
  echo "    ⚠ 未检测到 ffprobe！视频帧数核对/对齐检查会跳过（可先正常使用，稍后 sudo apt install ffmpeg）"
fi

echo "==> 4/4 配置与自检..."
if [ -f config.yaml ]; then
  if "$PYTHON_BIN" tools/check_config.py; then
    echo "    配置检查 ✓"
  else
    echo "    ⚠ 配置有问题（见上），可先不管继续，但建议修复后再用"
  fi
else
  echo "    尚无 config.yaml，第一次用请: cp config.example.yaml config.yaml  然后按需修改 paths 三个路径"
  echo "    （其余字段保持默认即可，见 README『配置速查表』）"
fi

echo ""
echo "==> 完成（环境: $MODE）。下一步："
echo "    bash tools/self_test.sh        # 一键自测（假数据全链路，推荐先跑）"
echo "    python3 run.py                 # 总入口：菜单式操作（盘点/时间戳/清洗/合并/登记/汇总）"
echo "    或直接调脚本: python3 pipe/01_inspect.py --help"
echo ""
echo "    提示：登记时机=数据处理达标后（05 合并 / 06 转换之后）再跑 ledger/record.py，"
echo "          详细见 README『登记』一节。"