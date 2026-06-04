# 历史回测引擎：T+1、手续费、止损、资金管理

import os
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    INIT_CAPITAL, COMMISSION_BUY, COMMISSION_SELL,
    WINDOW_SIZE, BUY_THRESHOLD, SELL_THRESHOLD,
    STOP_LOSS_RATIO, MAX_POSITION_RATIO, MAX_HOLDINGS,
    REPORTS_DIR,
)
from models.lstm_model import LSTMModel
from models.trainer import predict_proba
from features.builder import build_inference_sequence
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    历史回测引擎：严格T+1、手续费、单股止损。

    初始资金 100万，买入0.03%/卖出0.13% 手续费。
    """

    def __init__(self, stock_data: dict, model: LSTMModel, scaler: MinMaxScaler):
        self.stock_data = stock_data
        self.model      = model
        self.scaler     = scaler
        self.cash       = float(INIT_CAPITAL)
        self.holdings   = {}      # {code: {shares, cost, buy_date}}
        self.today_bought = set()
        self.nav_curve  = []
        self.trades     = []

    def _get_price(self, code: str, date: pd.Timestamp) -> float:
        df  = self.stock_data.get(code)
        if df is None:
            return 0.0
        row = df[df["date"] == date]
        return float(row["close"].iloc[0]) if not row.empty else 0.0

    def _portfolio_value(self, date: pd.Timestamp) -> float:
        total = self.cash
        for code, pos in self.holdings.items():
            price = self._get_price(code, date)
            if price > 0:
                total += pos["shares"] * price
        return total

    def _buy(self, code: str, price: float, date: pd.Timestamp, total_value: float):
        max_amount = total_value * MAX_POSITION_RATIO
        shares     = int(max_amount / price / 100) * 100
        if shares <= 0:
            return
        cost = shares * price * (1 + COMMISSION_BUY)
        if cost > self.cash:
            shares = int(self.cash / (price * (1 + COMMISSION_BUY)) / 100) * 100
            cost   = shares * price * (1 + COMMISSION_BUY)
        if shares <= 0:
            return

        self.cash -= cost
        self.holdings[code] = {"shares": shares, "cost": price, "buy_date": date}
        self.today_bought.add(code)
        self.trades.append({
            "date": date, "code": code, "action": "buy",
            "price": price, "shares": shares,
            "commission": shares * price * COMMISSION_BUY,
        })

    def _sell(self, code: str, price: float, date: pd.Timestamp, reason: str = "signal"):
        if code not in self.holdings:
            return
        shares   = self.holdings[code]["shares"]
        proceeds = shares * price * (1 - COMMISSION_SELL)
        self.cash += proceeds
        self.trades.append({
            "date": date, "code": code, "action": "sell",
            "price": price, "shares": shares,
            "commission": shares * price * COMMISSION_SELL,
            "reason": reason,
        })
        del self.holdings[code]

    def run(self) -> dict:
        """执行完整回测，返回绩效指标字典。"""
        from backtest.metrics import compute_metrics

        all_dates = set()
        for df in self.stock_data.values():
            all_dates.update(df["date"].tolist())
        all_dates = sorted(all_dates)

        logger.info(f"回测区间: {all_dates[0].date()} ~ {all_dates[-1].date()}")

        for date in all_dates:
            self.today_bought = set()

            # 构建当日信号
            signals = {}
            for code, df in self.stock_data.items():
                sub = df[df["date"] <= date]
                if len(sub) < WINDOW_SIZE + 1:
                    continue
                try:
                    X    = build_inference_sequence(sub, self.scaler)
                    prob = float(predict_proba(self.model, X)[0])
                    signals[code] = prob
                except Exception:
                    pass

            total_value = self._portfolio_value(date)

            # 单股止损（T+1：今日买入不卖）
            for code in list(self.holdings.keys()):
                if code in self.today_bought:
                    continue
                price = self._get_price(code, date)
                if price <= 0:
                    continue
                pnl = (price - self.holdings[code]["cost"]) / self.holdings[code]["cost"]
                if pnl < STOP_LOSS_RATIO:
                    self._sell(code, price, date, reason="stop_loss")

            # 卖出信号
            for code in list(self.holdings.keys()):
                if code in self.today_bought:
                    continue
                if signals.get(code, 0.5) < SELL_THRESHOLD:
                    price = self._get_price(code, date)
                    if price > 0:
                        self._sell(code, price, date, reason="signal")

            # 买入信号
            for code, prob in sorted(signals.items(), key=lambda x: -x[1]):
                if len(self.holdings) >= MAX_HOLDINGS:
                    break
                if code in self.holdings:
                    continue
                if prob > BUY_THRESHOLD:
                    price = self._get_price(code, date)
                    if price > 0:
                        self._buy(code, price, date, total_value)

            self.nav_curve.append((date, self._portfolio_value(date)))

        nav_df  = pd.DataFrame(self.nav_curve, columns=["date", "nav"])
        metrics = compute_metrics(nav_df, self.trades)
        self._save_report(nav_df, metrics)
        return metrics

    def _save_report(self, nav_df: pd.DataFrame, metrics: dict):
        """保存净值曲线图。"""
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(nav_df["date"], nav_df["nav"] / INIT_CAPITAL, linewidth=1.5, label="AlphaQuant")
        ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="基准线")
        ax.set_title("AlphaQuant 回测净值曲线")
        ax.set_ylabel("净值")
        ax.set_xlabel("日期")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()

        path = os.path.join(REPORTS_DIR, "backtest_nav.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info(f"报告已保存: {path}")

        print("\n" + "=" * 50)
        print("  AlphaQuant 回测绩效报告")
        print("=" * 50)
        for k, v in metrics.items():
            print(f"  {k:<16} {v}")
        print("=" * 50)
