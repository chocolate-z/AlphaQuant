#!/bin/bash
echo "============================================"
echo "  AlphaQuant 一键安装脚本（Mac / Linux）"
echo "============================================"

if ! command -v python3 &>/dev/null; then
    echo "[错误] 未检测到 Python3，请先安装"
    exit 1
fi

echo "[1/3] 检测到 $(python3 --version)"

if [ ! -d "venv" ]; then
    echo "[2/3] 正在创建虚拟环境..."
    python3 -m venv venv
else
    echo "[2/3] 虚拟环境已存在，跳过"
fi

echo "[3/3] 正在安装依赖（首次约需 5~10 分钟）..."
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q

echo ""
echo "============================================"
echo "  安装完成！运行以下命令启动："
echo "  source venv/bin/activate && python main.py"
echo "============================================"
