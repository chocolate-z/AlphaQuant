# 命令行主入口：train / backtest / paper / dashboard / signal / diagnose

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


def cmd_train(args):
    from data.loader import load_all_stocks
    from features.builder import build_all_stocks
    from models.trainer import train_model

    logger.info("=== 模式：训练 ===")
    logger.info("正在下载数据...")
    stock_data = load_all_stocks(force_refresh=getattr(args, "refresh", False))

    if not stock_data:
        logger.error("未获取到任何数据，请检查网络和 AKShare 版本")
        sys.exit(1)

    logger.info("正在构建特征...")
    X, y, scaler = build_all_stocks(stock_data, fit_scaler=True)

    if len(X) == 0:
        logger.error("特征构建失败，数据可能不足")
        sys.exit(1)

    logger.info(f"特征维度: {X.shape}，正样本比例: {y.mean():.3f}")
    logger.info("开始训练模型...")
    train_model(X, y)
    logger.info("训练完成！")


def cmd_backtest(args):
    from data.loader import load_all_stocks
    from features.builder import load_scaler
    from models.lstm_model import load_model
    from backtest.engine import BacktestEngine
    from config import MODEL_SAVE_DIR

    logger.info("=== 模式：回测 ===")
    model_path = os.path.join(MODEL_SAVE_DIR, "lstm_best.pt")

    if not os.path.exists(model_path):
        logger.error("模型文件不存在，请先运行 --mode train")
        sys.exit(1)

    stock_data = load_all_stocks()
    model      = load_model(model_path)
    scaler     = load_scaler()

    engine  = BacktestEngine(stock_data, model, scaler)
    metrics = engine.run()
    logger.info("回测完成")


def cmd_paper(args):
    logger.info("=== 模式：模拟盘 ===")
    from paper_trading.scheduler import start_scheduler
    start_scheduler()


def cmd_dashboard(args):
    logger.info("=== 模式：看板 ===")
    from dashboard.app import start_dashboard
    start_dashboard()


def cmd_signal(args):
    logger.info("=== 模式：信号 ===")
    from paper_trading.executor import get_current_signals
    from data.loader import get_stock_name
    from config import BUY_THRESHOLD, SELL_THRESHOLD

    signals = get_current_signals()
    if not signals:
        logger.error("未获取到信号，请检查模型是否已训练")
        return

    print(f"\n{'代码':<14} {'名称':<8} {'买入概率':>10} {'建议':>10}")
    print("-" * 50)
    for code, prob in sorted(signals.items(), key=lambda x: -x[1]):
        name = get_stock_name(code)[:6]
        sug  = "★买入" if prob > BUY_THRESHOLD else ("▼观望" if prob < SELL_THRESHOLD else "─持仓")
        print(f"{code:<14} {name:<8} {prob:>10.1%} {sug:>10}")
    print()


def cmd_diagnose(args):
    logger.info("=== 模式：诊断 ===")
    from diagnose.analyzer import diagnose
    from diagnose.report import print_report, print_batch_summary

    if not args.stock:
        logger.error("请通过 --stock 指定股票代码，如 --stock sh600519")
        sys.exit(1)

    codes = [s.strip() for s in args.stock.split(",")]
    cost  = float(args.cost) if getattr(args, "cost", None) else None

    if len(codes) == 1:
        try:
            result = diagnose(codes[0], holdings_cost=cost)
            print_report(result)
        except Exception as e:
            logger.error(f"诊断失败: {e}")
    else:
        results = []
        for code in codes:
            try:
                r = diagnose(code, holdings_cost=cost)
                print_report(r)
                results.append(r)
            except Exception as e:
                logger.error(f"[{code}] 诊断失败: {e}")
        if results:
            print_batch_summary(results)


# ── 入口 ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AlphaQuant — A股量化AI系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例：
  python main.py --mode train              # 下载数据并训练模型
  python main.py --mode train --refresh    # 强制重新下载所有数据
  python main.py --mode backtest           # 运行历史回测
  python main.py --mode paper              # 启动模拟盘调度器（阻塞）
  python main.py --mode dashboard          # 启动网页看板
  python main.py --mode signal             # 查看今日信号排行
  python main.py --mode diagnose --stock sh600519
  python main.py --mode diagnose --stock sh600519,sz000858,sz300750
  python main.py --mode diagnose --stock sh600519 --cost 1650.00
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["train", "backtest", "paper", "dashboard", "signal", "diagnose"],
        required=True,
        help="运行模式",
    )
    parser.add_argument("--stock",   type=str, default=None,
                        help="股票代码，多只用逗号分隔（diagnose 模式）")
    parser.add_argument("--cost",    type=str, default=None,
                        help="持仓成本价（diagnose 模式，用于止盈止损判断）")
    parser.add_argument("--refresh", action="store_true",
                        help="强制重新下载数据（train 模式）")

    args = parser.parse_args()

    dispatch = {
        "train":     cmd_train,
        "backtest":  cmd_backtest,
        "paper":     cmd_paper,
        "dashboard": cmd_dashboard,
        "signal":    cmd_signal,
        "diagnose":  cmd_diagnose,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()
