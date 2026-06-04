# 交易日志：每笔交易 CSV + 每日账户快照（防重复写入）

import os
import csv
import logging
from datetime import datetime

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LOGS_DIR, INIT_CAPITAL, COMMISSION_BUY, COMMISSION_SELL

logger = logging.getLogger(__name__)

DAILY_SNAPSHOT_FILE = os.path.join(LOGS_DIR, "account_daily.csv")


class TradeLogger:
    """记录交易明细和每日账户快照。"""

    def log_trade(self, code: str, action: str, price: float, shares: int,
                  reason: str = "", commission: float = None):
        """
        记录单笔交易到当日 CSV（logs/trades_YYYYMMDD.csv）。
        """
        today       = datetime.today().strftime("%Y%m%d")
        file_path   = os.path.join(LOGS_DIR, f"trades_{today}.csv")
        file_exists = os.path.exists(file_path)

        if commission is None:
            rate       = COMMISSION_BUY if action == "buy" else COMMISSION_SELL
            commission = shares * price * rate

        with open(file_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["time", "code", "action", "price", "shares", "commission", "reason"])
            writer.writerow([
                datetime.now().strftime("%H:%M:%S"),
                code, action, f"{price:.2f}", shares, f"{commission:.2f}", reason,
            ])

    def log_daily_snapshot(self, account):
        """
        记录每日账户快照到 account_daily.csv。
        若当天已有记录则更新，防止重复触发时重复写入。
        """
        today       = datetime.today().strftime("%Y-%m-%d")
        total       = account.total_assets
        holding_val = account.holding_value
        cum_pnl     = total - INIT_CAPITAL

        # 读取所有现有记录
        rows = []
        headers = ["date", "total_assets", "cash", "holding_value", "daily_pnl", "cum_pnl"]
        if os.path.exists(DAILY_SNAPSHOT_FILE):
            try:
                with open(DAILY_SNAPSHOT_FILE, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    all_rows = list(reader)
                if all_rows:
                    headers = all_rows[0]
                    rows    = all_rows[1:]
            except Exception as e:
                logger.error(f"读取日快照失败: {e}")

        # 计算当日盈亏（与昨日总资产对比）
        daily_pnl = 0.0
        if rows:
            try:
                last_row = rows[-1]
                if last_row[0] != today:  # 昨日数据才用于计算日盈亏
                    daily_pnl = total - float(last_row[1])
            except Exception:
                pass

        new_row = [today, f"{total:.2f}", f"{account.cash:.2f}",
                   f"{holding_val:.2f}", f"{daily_pnl:.2f}", f"{cum_pnl:.2f}"]

        # 如果今天已有记录则更新，否则追加
        if rows and rows[-1][0] == today:
            rows[-1] = new_row
        else:
            rows.append(new_row)

        # 写回文件
        with open(DAILY_SNAPSHOT_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)

        logger.info(f"日快照已更新：总资产 ¥{total:,.0f}，当日盈亏 ¥{daily_pnl:,.0f}，累计 ¥{cum_pnl:,.0f}")
