# 风控模块：单股止损、组合止损（峰值回撤）、仓位限制

import os
import json
import logging
from datetime import date, timedelta

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import STOP_LOSS_RATIO, PORTFOLIO_STOP, SUSPEND_DAYS, LOGS_DIR

logger = logging.getLogger(__name__)

SUSPEND_FILE     = os.path.join(LOGS_DIR, "suspend_state.json")
PEAK_ASSETS_FILE = os.path.join(LOGS_DIR, "peak_assets.json")


class RiskManager:
    """
    风控管理器。
    组合止损基于从历史最高净值的回撤计算，而非从初始资金计算。
    """

    def __init__(self, account):
        self.account      = account
        self._peak_assets = self._load_peak()

    # ── 峰值跟踪 ──────────────────────────────────────

    def _load_peak(self) -> float:
        if os.path.exists(PEAK_ASSETS_FILE):
            try:
                with open(PEAK_ASSETS_FILE) as f:
                    return float(json.load(f).get("peak", self.account.total_assets))
            except Exception:
                pass
        return self.account.total_assets

    def _save_peak(self):
        with open(PEAK_ASSETS_FILE, "w") as f:
            json.dump({"peak": self._peak_assets, "date": date.today().isoformat()}, f)

    def update_peak(self):
        """每日更新历史最高净值。"""
        current = self.account.total_assets
        if current > self._peak_assets:
            self._peak_assets = current
            self._save_peak()
            logger.info(f"新高净值: ¥{current:,.0f}")

    @property
    def current_drawdown(self) -> float:
        """当前从峰值的回撤比例（负数）。"""
        return (self.account.total_assets - self._peak_assets) / (self._peak_assets + 1e-9)

    # ── 止损检查 ──────────────────────────────────────

    def check_single_stop_loss(self, code: str, current_price: float) -> bool:
        """
        单股止损检查：持仓亏损超过 STOP_LOSS_RATIO 则触发。

        Returns:
            True 表示需要止损卖出
        """
        pos = self.account.holdings.get(code)
        if pos is None:
            return False
        pnl = (current_price - pos["cost"]) / pos["cost"]
        if pnl < STOP_LOSS_RATIO:
            logger.warning(f"[{code}] 触发单股止损：浮亏 {pnl*100:.1f}%，成本 {pos['cost']:.2f}，现价 {current_price:.2f}")
            return True
        return False

    def check_portfolio_stop(self) -> bool:
        """
        组合止损检查：总资产从历史最高净值回撤超过 PORTFOLIO_STOP 则触发。
        （修复：之前错误地从初始资金计算，现在正确地从峰值计算）

        Returns:
            True 表示触发组合止损
        """
        self.update_peak()
        dd = self.current_drawdown
        if dd < PORTFOLIO_STOP:
            logger.warning(f"触发组合止损：从峰值(¥{self._peak_assets:,.0f})回撤 {dd*100:.1f}%")
            self._set_suspend()
            return True
        return False

    def _set_suspend(self):
        state = {
            "suspend_until": (date.today() + timedelta(days=SUSPEND_DAYS)).isoformat(),
            "reason":        f"组合回撤 {self.current_drawdown*100:.1f}%",
        }
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
            if date.today() < until:
                days_left = (until - date.today()).days
                logger.info(f"模拟盘暂停中，还剩 {days_left} 个交易日（到 {until}）")
                return True
            # 暂停期结束，清除文件
            os.remove(SUSPEND_FILE)
            return False
        except Exception:
            return False
