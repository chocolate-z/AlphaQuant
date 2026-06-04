# 交易日志：每笔交易 CSV + 每日账户快照

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

        Args:
            code: 股票代码
            action: 'buy' 或 'sell'
            price: 成交价
            shares: 股数
            reason: 触发原因
            commission: 手续费（None 则自动计算）
        """
        today     = datetime.today().strftime("%Y%m%d")
        file_path = os.path.join(LOGS_DIR, f"trades_{today}.csv")
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

        Args:
            account: VirtualAccount 实例
        """
        today       = datetime.today().strftime("%Y-%m-%d")
        total       = account.total_assets
        holding_val = account.holding_value
        cum_pnl     = total - INIT_CAPITAL

        daily_pnl = 0.0
        if os.path.exists(DAILY_SNAPSHOT_FILE):
            try:
                with open(DAILY_SNAPSHOT_FILE, "r", encoding="utf-8") as f:
                    rows = list(csv.reader(f))
                if len(rows) > 1:
                    last_total = float(rows[-1][1])
                    daily_pnl  = total - last_total
            except Exception:
                pass

        file_exists = os.path.exists(DAILY_SNAPSHOT_FILE)
        with open(DAILY_SNAPSHOT_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["date", "total_assets", "cash", "holding_value", "daily_pnl", "cum_pnl"])
            writer.writerow([
                today,
                f"{total:.2f}", f"{account.cash:.2f}",
                f"{holding_val:.2f}", f"{daily_pnl:.2f}", f"{cum_pnl:.2f}",
            ])
        logger.info(f"日快照已记录：总资产 ¥{total:,.0f}，当日盈亏 ¥{daily_pnl:,.0f}")
