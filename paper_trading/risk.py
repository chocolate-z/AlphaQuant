# 风控模块：单股止损、组合止损、仓位限制

import os
import json
import logging
from datetime import date, timedelta

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import STOP_LOSS_RATIO, PORTFOLIO_STOP, SUSPEND_DAYS, INIT_CAPITAL, LOGS_DIR

logger = logging.getLogger(__name__)

SUSPEND_FILE = os.path.join(LOGS_DIR, "suspend_state.json")


class RiskManager:
    """风控管理器，依赖 VirtualAccount 实例。"""

    def __init__(self, account):
        self.account = account

    def check_single_stop_loss(self, code: str, current_price: float) -> bool:
        """
        单股止损检查。

        Returns:
            True 表示需要止损卖出
        """
        pos = self.account.holdings.get(code)
        if pos is None:
            return False
        pnl = (current_price - pos["cost"]) / pos["cost"]
        if pnl < STOP_LOSS_RATIO:
            logger.warning(f"[{code}] 触发单股止损：浮亏 {pnl*100:.1f}%")
            return True
        return False

    def check_portfolio_stop(self) -> bool:
        """
        组合止损检查：总资产从初始资金回撤超过 -12% 则触发。

        Returns:
            True 表示触发组合止损
        """
        total    = self.account.total_assets
        drawdown = (total - INIT_CAPITAL) / INIT_CAPITAL
        if drawdown < PORTFOLIO_STOP:
            logger.warning(f"触发组合止损：总回撤 {drawdown*100:.1f}%")
            self._set_suspend()
            return True
        return False

    def _set_suspend(self):
        state = {"suspend_until": (date.today() + timedelta(days=SUSPEND_DAYS)).isoformat()}
        with open(SUSPEND_FILE, "w") as f:
            json.dump(state, f)

    def is_suspended(self) -> bool:
        """是否处于交易暂停状态。"""
        if not os.path.exists(SUSPEND_FILE):
            return False
        try:
            with open(SUSPEND_FILE) as f:
                state = json.load(f)
            until = date.fromisoformat(state["suspend_until"])
            return date.today() < until
        except Exception:
            return False
