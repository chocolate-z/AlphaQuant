@echo off
chcp 65001 >nul
echo ============================================
echo   AlphaQuant 一键安装脚本（Windows）
echo ============================================
echo.

:: 检查 Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到 Python，请先安装 Python 3.10 或以上版本
    echo 下载地址：https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [1/3] 检测到 Python：
python --version
echo.

:: 创建虚拟环境
if not exist "venv" (
    echo [2/3] 正在创建虚拟环境...
    python -m venv venv
) else (
    echo [2/3] 虚拟环境已存在，跳过创建
)

:: 安装依赖
echo [3/3] 正在安装依赖（首次约需 5~10 分钟）...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip -q
pip install -r requirements.txt -q

echo.
echo ============================================
echo   安装完成！运行 start.bat 启动程序
echo ============================================
pause
