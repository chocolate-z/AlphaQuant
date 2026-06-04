# 虚拟账户：资金、持仓管理，T+1限制，状态持久化

import os
import json
import logging
from datetime import datetime
from typing import Dict

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import INIT_CAPITAL, COMMISSION_BUY, COMMISSION_SELL, LOGS_DIR

logger = logging.getLogger(__name__)

STATE_FILE = os.path.join(LOGS_DIR, "account_state.json")


class VirtualAccount:
    """虚拟账户，管理现金、持仓和T+1约束。"""

    def __init__(self):
        self.cash: float          = float(INIT_CAPITAL)
        self.holdings: Dict[str, dict] = {}
        self.today_bought: set    = set()
        self.load_state()

    # ── 属性 ──────────────────────────────────────────

    @property
    def holding_value(self) -> float:
        return sum(
            p.get("market_value", p["shares"] * p["cost"])
            for p in self.holdings.values()
        )

    @property
    def total_assets(self) -> float:
        return self.cash + self.holding_value

    # ── 交易操作 ──────────────────────────────────────

    def buy(self, code: str, price: float, shares: int) -> bool:
        """
        买入股票。

        Returns:
            True 表示成功
        """
        if shares <= 0 or price <= 0:
            return False

        cost = shares * price * (1 + COMMISSION_BUY)
        if cost > self.cash:
            logger.warning(f"资金不足，买入 {code} 失败（需 {cost:.0f}，有 {self.cash:.0f}）")
            return False

        self.cash -= cost
        if code in self.holdings:
            old          = self.holdings[code]
            total_shares = old["shares"] + shares
            avg_cost     = (old["shares"] * old["cost"] + shares * price) / total_shares
            self.holdings[code]["shares"] = total_shares
            self.holdings[code]["cost"]   = avg_cost
        else:
            self.holdings[code] = {
                "shares":       shares,
                "cost":         price,
                "buy_date":     datetime.today().strftime("%Y-%m-%d"),
                "market_value": shares * price,
                "current_price": price,
            }
        self.today_bought.add(code)
        logger.info(f"买入 {code}: {shares}股 @{price:.2f}")
        self.save_state()
        return True

    def sell(self, code: str, price: float, shares: int = None) -> bool:
        """
        卖出股票（shares=None 表示全部卖出）。

        Returns:
            True 表示成功
        """
        if code not in self.holdings:
            logger.warning(f"无持仓，无法卖出 {code}")
            return False
        if code in self.today_bought:
            logger.warning(f"T+1 限制，今日买入的 {code} 不可卖出")
            return False

        pos = self.holdings[code]
        if shares is None:
            shares = pos["shares"]
        shares = min(shares, pos["shares"])

        proceeds  = shares * price * (1 - COMMISSION_SELL)
        self.cash += proceeds

        if shares >= pos["shares"]:
            del self.holdings[code]
        else:
            self.holdings[code]["shares"] -= shares

        logger.info(f"卖出 {code}: {shares}股 @{price:.2f}")
        self.save_state()
        return True

    def update_prices(self, price_dict: Dict[str, float]):
        """更新持仓市值（每日收盘后调用）。"""
        for code in self.holdings:
            price = price_dict.get(code, self.holdings[code]["cost"])
            self.holdings[code]["market_value"]  = self.holdings[code]["shares"] * price
            self.holdings[code]["current_price"] = price

    def new_day(self):
        """每日开始时清除 T+1 标记。"""
        self.today_bought = set()
        self.save_state()

    # ── 持久化 ────────────────────────────────────────

    def save_state(self):
        state = {
            "cash":         self.cash,
            "holdings":     self.holdings,
            "today_bought": list(self.today_bought),
            "updated_at":   datetime.now().isoformat(),
        }
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def load_state(self):
        if not os.path.exists(STATE_FILE):
            self.save_state()
            return
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                state = json.load(f)
            self.cash         = float(state.get("cash", INIT_CAPITAL))
            self.holdings     = state.get("holdings", {})
            self.today_bought = set(state.get("today_bought", []))
            logger.info(f"账户已恢复，总资产 ¥{self.total_assets:,.0f}")
        except Exception as e:
            logger.error(f"账户状态加载失败: {e}，使用默认值")
