#!/usr/bin/env bash
# AI 大脑一键装配（公司机 Linux + GPU 首选）
# 用法:  bash tools/setup_ai.sh [模型名]
# 默认模型 qwen2.5:14b（适配 RTX 3090 24G）；显存小可换 qwen2.5:7b
# 装完后在数据工作台里：点顶栏 AI → 选「本地 Ollama」→ 接口自动填，模型名填下面用的模型，Key 留空
set -e

MODEL="${1:-qwen2.5:14b}"

echo "==> [1/3] 检查 Ollama"
if command -v ollama >/dev/null 2>&1; then
  echo "    已安装: $(ollama --version 2>/dev/null | head -1 || echo ollama)"
else
  echo "    未安装，开始安装（官方脚本，需要 sudo 权限；无 sudo 请手动装后重跑）"
  curl -fsSL https://ollama.com/install.sh | sh
fi

echo "==> [2/3] 拉取模型 $MODEL（首次会下载数 GB，视网速等待）"
ollama pull "$MODEL"

echo "==> [3/3] 冒烟测试"
ollama run "$MODEL" "只回复两个字：在的"

echo ""
echo "✔ 装配完成。数据工作台 AI 设置填入："
echo "   引擎 = 本地 Ollama"
echo "   接口 = http://127.0.0.1:11434/v1"
echo "   模型 = $MODEL"
echo "   Key  = 留空"
echo ""
echo "提示：家里那台（无 GPU）可装 Ollama 后改用小模型：ollama pull qwen2.5:3b"
