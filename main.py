# AlphaQuant 主入口：交互式菜单 + 命令行两种使用方式

import argparse
import logging
import sys
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

# ── 菜单工具 ─────────────────────────────────────────────────────────

_W = 50  # 菜单宽度（字节宽，中文占2字节）

def _box_top():  print("╔" + "═" * (_W - 2) + "╗")
def _box_sep():  print("╠" + "═" * (_W - 2) + "╣")
def _box_bot():  print("╚" + "═" * (_W - 2) + "╝")

def _box_line(s: str = ""):
    # 按字节宽度计算填充（中文2字节，英文1字节）
    byte_w = len(s.encode("utf-8", errors="replace")) - len(s.encode("ascii", errors="ignore")) // 2
    try:
        from wcwidth import wcswidth
        display_w = wcswidth(s)
        display_w = display_w if display_w >= 0 else len(s)
    except ImportError:
        display_w = sum(2 if ord(c) > 127 else 1 for c in s)
    pad = _W - 4 - display_w
    print(f"║  {s}{' ' * max(pad, 0)}║")

def _clear():
    os.system("cls" if os.name == "nt" else "clear")

def _pause():
    input("\n  按 Enter 返回菜单...")

def _ask(prompt: str, default: str = "") -> str:
    tip = f"（回车默认 {default}）" if default else ""
    val = input(f"  {prompt}{tip}：").strip()
    return val if val else default

def _confirm(prompt: str) -> bool:
    return _ask(prompt + " (yes/no)", "no").lower() == "yes"


def _print_menu():
    _clear()
    _box_top()
    _box_line()
    _box_line("    AlphaQuant — A股量化 AI 系统")
    _box_line()
    _box_sep()
    _box_line("  【训练 & 回测】")
    _box_line("  1  训练模型（完整模式，全部股票）")
    _box_line("  2  训练模型（快速模式，随机10只/近2年）")
    _box_line("  3  历史回测")
    _box_sep()
    _box_line("  【实盘 & 看板】")
    _box_line("  4  启动模拟盘调度器")
    _box_line("  5  启动网页看板")
    _box_line("  6  查看今日信号排行")
    _box_sep()
    _box_line("  【工具】")
    _box_line("  7  个股诊断")
    _box_line("  8  查看训练/回测报告图表")
    _box_line("  9  重置虚拟账户")
    _box_line()
    _box_line("  0  退出")
    _box_bot()
    print()


# ── 各功能 ───────────────────────────────────────────────────────────

def run_train(quick: bool = False, force_refresh: bool = False):
    from data.loader import load_all_stocks
    from features.builder import build_all_stocks
    from models.trainer import train_model

    mode_tag = "快速模式" if quick else "完整模式"
    logger.info(f"=== 训练模型 [{mode_tag}] ===")

    logger.info("正在下载/加载数据...")
    stock_data = load_all_stocks(force_refresh=force_refresh, quick=quick)

    if not stock_data:
        logger.error("未获取到任何数据，请检查网络连接")
        return

    logger.info(f"共加载 {len(stock_data)} 只股票，正在构建特征...")
    X, y, scaler = build_all_stocks(stock_data, fit_scaler=True)

    if len(X) == 0:
        logger.error("特征构建失败，数据不足（每只股票至少需要 40 条记录）")
        return

    logger.info(f"特征维度: {X.shape}，正样本比例: {y.mean():.3f}")
    logger.info("开始训练...")
    train_model(X, y)
    logger.info("训练完成！")


def run_backtest():
    from data.loader import load_all_stocks
    from features.builder import load_scaler
    from models.lstm_model import load_model
    from backtest.engine import BacktestEngine
    from data.index_fetcher import fetch_all_benchmarks
    from config import MODEL_SAVE_DIR, START_DATE, RELATIVE_RANK_MODE

    model_path = os.path.join(MODEL_SAVE_DIR, "lstm_best.pt")
    if not os.path.exists(model_path):
        logger.error("模型文件不存在，请先训练模型（选项 1 或 2）")
        return

    mode_tag = "相对排名模式" if RELATIVE_RANK_MODE else "绝对阈值模式"
    logger.info(f"=== 历史回测 [{mode_tag}] ===")

    from datetime import datetime
    end_str = datetime.today().strftime("%Y%m%d")
    start_str = START_DATE.replace("-", "")

    logger.info("正在拉取5个基准指数历史数据...")
    benchmarks = fetch_all_benchmarks(start_str, end_str)
    if benchmarks:
        logger.info(f"成功加载基准指数: {list(benchmarks.keys())}")
    else:
        logger.warning("基准指数拉取失败，将跳过对比图")

    stock_data = load_all_stocks()
    model      = load_model(model_path)
    scaler     = load_scaler()
    BacktestEngine(stock_data, model, scaler, benchmarks=benchmarks).run()
    logger.info("回测完成")


def run_paper():
    logger.info("=== 模拟盘调度器（Ctrl+C 停止）===")
    from paper_trading.scheduler import start_scheduler
    start_scheduler()


def run_dashboard():
    from config import DASHBOARD_HOST, DASHBOARD_PORT
    logger.info("=== 网页看板 ===")
    print(f"\n  看板地址：http://{DASHBOARD_HOST}:{DASHBOARD_PORT}")
    print("  在浏览器中打开上述地址，Ctrl+C 停止\n")
    from dashboard.app import start_dashboard
    start_dashboard()


def run_signal():
    from paper_trading.executor import get_current_signals
    from data.loader import get_stock_name
    from config import BUY_THRESHOLD, SELL_THRESHOLD

    logger.info("=== 今日信号排行 ===")
    signals = get_current_signals()
    if not signals:
        logger.error("未获取到信号，请先训练模型并运行一次模拟盘")
        return

    print(f"\n  {'代码':<14} {'名称':<8} {'买入概率':>10} {'建议':>10}")
    print("  " + "─" * 48)
    for code, prob in sorted(signals.items(), key=lambda x: -x[1]):
        name = get_stock_name(code)[:6]
        sug  = "★ 买入" if prob > BUY_THRESHOLD else ("▼ 观望" if prob < SELL_THRESHOLD else "─ 持仓")
        print(f"  {code:<14} {name:<8} {prob:>10.1%} {sug:>10}")
    print()


def run_diagnose(stock_input: str = None, cost: float = None):
    from diagnose.analyzer import diagnose
    from diagnose.report import print_report, print_batch_summary

    if stock_input is None:
        print()
        stock_input = _ask("输入股票代码（多只用逗号分隔，如 sh600519,sz000858）")
        if not stock_input:
            print("  已取消")
            return
        cost_str = _ask("持仓成本价（无则直接回车）")
        cost = float(cost_str) if cost_str else None

    codes = [s.strip() for s in stock_input.split(",") if s.strip()]
    results = []
    for code in codes:
        try:
            r = diagnose(code, holdings_cost=cost)
            print_report(r)
            results.append(r)
        except Exception as e:
            logger.error(f"[{code}] 诊断失败: {e}")
    if len(results) > 1:
        print_batch_summary(results)


def run_view_reports():
    """列出并用系统默认程序打开已生成的报告图表。"""
    from config import REPORTS_DIR
    import glob

    pngs = sorted(glob.glob(os.path.join(REPORTS_DIR, "*.png")))
    if not pngs:
        print("\n  暂无报告文件，请先运行训练或回测\n")
        return

    print("\n  已生成的报告文件：")
    for i, p in enumerate(pngs, 1):
        size_kb = os.path.getsize(p) // 1024
        print(f"  {i}. {os.path.basename(p)}  ({size_kb} KB)  {p}")

    choice = _ask(f"\n  输入编号打开（1-{len(pngs)}），直接回车全部打开", "all")
    import subprocess, sys as _sys

    def _open(path):
        try:
            if os.name == "nt":
                os.startfile(path)
            elif _sys.platform == "darwin":
                subprocess.Popen(["open", path])
            else:
                subprocess.Popen(["xdg-open", path])
            print(f"  已打开: {os.path.basename(path)}")
        except Exception as e:
            print(f"  无法自动打开，请手动查看: {path}")

    if choice == "all":
        for p in pngs:
            _open(p)
    elif choice.isdigit() and 1 <= int(choice) <= len(pngs):
        _open(pngs[int(choice) - 1])
    else:
        print("  无效输入")


def run_reset():
    import json
    from config import LOGS_DIR, INIT_CAPITAL

    state_file   = os.path.join(LOGS_DIR, "account_state.json")
    suspend_file = os.path.join(LOGS_DIR, "suspend_state.json")
    peak_file    = os.path.join(LOGS_DIR, "peak_assets.json")

    print()
    print(f"  ⚠️  此操作将重置虚拟账户：")
    print(f"     · 现金恢复为 ¥{INIT_CAPITAL:,.0f}")
    print(f"     · 清空所有持仓、暂停状态和峰值记录")
    print()

    if not _confirm("确认重置"):
        print("  已取消")
        return

    new_state = {
        "cash": float(INIT_CAPITAL),
        "holdings": {},
        "today_bought": [],
        "updated_at": __import__("datetime").datetime.now().isoformat(),
    }
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(new_state, f, ensure_ascii=False, indent=2)

    for fp in [suspend_file, peak_file]:
        if os.path.exists(fp):
            os.remove(fp)

    print(f"\n  ✓ 账户已重置，初始资金 ¥{INIT_CAPITAL:,.0f}")


# ── 交互式菜单主循环 ──────────────────────────────────────────────────

def interactive_menu():
    handlers = {
        "1": lambda: run_train(quick=False, force_refresh=_confirm("是否强制重新下载数据")),
        "2": lambda: (_clear(), print("\n  快速模式：随机抽取 10 只股票，近 2 年历史\n"), run_train(quick=True)),
        "3": run_backtest,
        "4": run_paper,
        "5": run_dashboard,
        "6": run_signal,
        "7": run_diagnose,
        "8": run_view_reports,
        "9": run_reset,
    }

    while True:
        _print_menu()
        choice = input("  请输入选项编号：").strip()

        if choice == "0":
            print("\n  再见！\n")
            break

        if choice not in handlers:
            print("  无效选项，请重新输入")
            import time; time.sleep(1)
            continue

        print()
        try:
            handlers[choice]()
        except KeyboardInterrupt:
            print("\n  已中断")
        except Exception as e:
            logger.error(f"执行失败: {e}")

        if choice not in ("4", "5", "8"):   # 这些选项操作完直接回菜单，不需要暂停
            _pause()


# ── 命令行模式（脚本 / 自动化使用）──────────────────────────────────────

def main():
    if len(sys.argv) == 1:
        interactive_menu()
        return

    parser = argparse.ArgumentParser(
        description="AlphaQuant — A股量化AI系统（无参数直接运行进入交互菜单）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python main.py --mode train\n"
            "  python main.py --mode train --quick\n"
            "  python main.py --mode train --refresh\n"
            "  python main.py --mode backtest\n"
            "  python main.py --mode diagnose --stock sh600519\n"
            "  python main.py --mode diagnose --stock sh600519 --cost 1650\n"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["train", "backtest", "paper", "dashboard", "signal", "diagnose", "reset"],
        required=True,
    )
    parser.add_argument("--stock",   type=str, default=None)
    parser.add_argument("--cost",    type=str, default=None)
    parser.add_argument("--refresh", action="store_true", help="强制重新下载数据（train）")
    parser.add_argument("--quick",   action="store_true", help="快速模式（train）")

    args = parser.parse_args()

    dispatch = {
        "train":     lambda: run_train(quick=args.quick, force_refresh=args.refresh),
        "backtest":  run_backtest,
        "paper":     run_paper,
        "dashboard": run_dashboard,
        "signal":    run_signal,
        "diagnose":  lambda: run_diagnose(
                         stock_input=args.stock,
                         cost=float(args.cost) if args.cost else None
                     ),
        "reset":     run_reset,
    }
    dispatch[args.mode]()


if __name__ == "__main__":
    main()
