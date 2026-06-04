# 自动调度：使用 schedule 库在每个交易日 15:30 触发执行

import logging
import time

import schedule

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DAILY_RUN_TIME
from data.realtime import is_trade_day

logger = logging.getLogger(__name__)


def _job():
    """每日定时任务入口。"""
    if not is_trade_day():
        logger.info(f"[{DAILY_RUN_TIME}] 非交易日，跳过")
        return
    logger.info(f"[{DAILY_RUN_TIME}] 触发每日执行...")
    try:
        from paper_trading.executor import run_daily_execution
        run_daily_execution()
    except Exception as e:
        logger.error(f"每日执行异常: {e}", exc_info=True)


def start_scheduler():
    """
    启动后台调度器（阻塞运行）。
    每个交易日 DAILY_RUN_TIME 自动触发一次，非交易日跳过。
    """
    schedule.every().day.at(DAILY_RUN_TIME).do(_job)
    logger.info(f"调度器已启动，每日 {DAILY_RUN_TIME} 触发")
    print(f"[AlphaQuant] 模拟盘调度器运行中，每日 {DAILY_RUN_TIME} 自动执行（Ctrl+C 停止）")

    while True:
        schedule.run_pending()
        time.sleep(30)
