# 历史回测引擎：预计算信号、T+1、手续费、涨跌停、止损、基准对比

import os
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm

def _setup_chinese_font():
    """自动检测并设置中文字体（Windows/macOS/Linux 均适用）。"""
    candidates = [
        "Microsoft YaHei", "SimHei", "SimSun", "FangSong",   # Windows
        "Heiti SC", "PingFang SC", "STHeiti", "STSong",       # macOS
        "WenQuanYi Micro Hei", "Noto Sans CJK SC",            # Linux
        "Arial Unicode MS",
    ]
    available = {f.name for f in _fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"]       = name
            plt.rcParams["axes.unicode_minus"] = False
            return
    # 找不到中文字体时改用英文标签（不报警告）
    plt.rcParams["axes.unicode_minus"] = False

_setup_chinese_font()

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
from features.builder import build_sequences, compute_raw_features, FEATURE_NAMES
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

# 科创板(688xxx)和创业板(sz3xxxx)涨跌停幅度 20%，其余 10%
def _price_limit(code: str) -> float:
    pure = code[2:]
    if pure.startswith("688") or (code.startswith("sz") and pure.startswith("3")):
        return 0.20
    return 0.10


class BacktestEngine:
    """
    历史回测引擎。

    改进：
    - 预计算所有信号（O(S*N) 而非 O(S*D*N)），速度提升 ~100x
    - 处理涨跌停（涨停无法买入，跌停无法卖出）
    - 支持沪深300基准对比
    """

    def __init__(self, stock_data: dict, model: LSTMModel, scaler: MinMaxScaler,
                 benchmark_df: pd.DataFrame = None):
        self.stock_data   = stock_data
        self.model        = model
        self.scaler       = scaler
        self.benchmark_df = benchmark_df  # 沪深300日线数据（可选）
        self.cash         = float(INIT_CAPITAL)
        self.holdings     = {}
        self.today_bought = set()
        self.nav_curve    = []
        self.trades       = []

    # ── 价格查询（带索引缓存）──────────────────────────

    def _build_price_index(self):
        """构建 {code: {date: {close, open, pct_change}}} 索引，避免重复过滤。"""
        self._price_idx = {}
        for code, df in self.stock_data.items():
            self._price_idx[code] = df.set_index("date").to_dict("index")

    def _get_close(self, code: str, date: pd.Timestamp) -> float:
        return self._price_idx.get(code, {}).get(date, {}).get("close", 0.0)

    def _get_prev_close(self, code: str, date: pd.Timestamp) -> float:
        """获取前一交易日收盘价（用于涨跌停判断）。"""
        df = self.stock_data.get(code)
        if df is None:
            return 0.0
        idx = df[df["date"] == date].index
        if len(idx) == 0 or idx[0] == 0:
            return 0.0
        return float(df.iloc[idx[0] - 1]["close"])

    def _is_limit_up(self, code: str, date: pd.Timestamp) -> bool:
        close      = self._get_close(code, date)
        prev_close = self._get_prev_close(code, date)
        if close <= 0 or prev_close <= 0:
            return False
        return (close - prev_close) / prev_close >= _price_limit(code) * 0.98

    def _is_limit_down(self, code: str, date: pd.Timestamp) -> bool:
        close      = self._get_close(code, date)
        prev_close = self._get_prev_close(code, date)
        if close <= 0 or prev_close <= 0:
            return False
        return (close - prev_close) / prev_close <= -_price_limit(code) * 0.98

    def _portfolio_value(self, date: pd.Timestamp) -> float:
        total = self.cash
        for code, pos in self.holdings.items():
            price = self._get_close(code, date)
            if price > 0:
                total += pos["shares"] * price
        return total

    # ── 交易执行 ──────────────────────────────────────

    def _buy(self, code: str, price: float, date: pd.Timestamp, total_value: float):
        if self._is_limit_up(code, date):
            logger.debug(f"[{code}] 涨停，无法买入")
            return
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
        if self._is_limit_down(code, date):
            logger.debug(f"[{code}] 跌停，无法卖出（{reason}）")
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

    # ── 信号预计算（核心优化）──────────────────────────

    def _precompute_signals(self) -> dict:
        """
        一次性预计算所有股票所有日期的信号。
        复杂度从 O(S*D*N) 降至 O(S*N)，速度提升约100x。

        Returns:
            {(code, date): probability}
        """
        from sklearn.preprocessing import MinMaxScaler as _MMS
        signal_table = {}

        for code, df in self.stock_data.items():
            if len(df) < WINDOW_SIZE + 5:
                continue
            try:
                X, _, _, dates = build_sequences(df, scaler=self.scaler, fit_scaler=False)
                if len(X) == 0:
                    continue
                probs = predict_proba(self.model, X)
                for d, p in zip(dates, probs):
                    signal_table[(code, pd.Timestamp(d))] = float(p)
            except Exception as e:
                logger.warning(f"[{code}] 预计算失败: {e}")

        logger.info(f"信号预计算完成：{len(signal_table)} 条（{len(self.stock_data)} 只股票）")

        # 概率分布诊断（帮助判断阈值是否合理）
        if signal_table:
            all_probs = list(signal_table.values())
            p_arr = np.array(all_probs)
            above_buy  = (p_arr > BUY_THRESHOLD).sum()
            above_half = (p_arr > 0.5).sum()
            logger.info(
                f"概率分布 — min={p_arr.min():.3f}  mean={p_arr.mean():.3f}  "
                f"max={p_arr.max():.3f}  >0.5: {above_half}条  >{BUY_THRESHOLD}: {above_buy}条"
            )
            if above_buy == 0:
                logger.warning(
                    f"⚠️  没有任何信号超过买入阈值 {BUY_THRESHOLD}，将产生 0 笔交易。"
                    f"建议：① 重新训练模型（更多数据/更多轮次）"
                    f"② 或在 config.py 中适当降低 BUY_THRESHOLD（当前 {BUY_THRESHOLD}）"
                )

        return signal_table

    # ── 主回测循环 ────────────────────────────────────

    def run(self) -> dict:
        """执行完整回测，返回绩效指标字典。"""
        from backtest.metrics import compute_metrics

        self._build_price_index()
        logger.info("正在预计算所有信号（可能需要1-2分钟）...")
        signal_table = self._precompute_signals()

        all_dates = set()
        for df in self.stock_data.values():
            all_dates.update(df["date"].tolist())
        all_dates = sorted(all_dates)

        logger.info(f"回测区间: {all_dates[0].date()} ~ {all_dates[-1].date()}，共 {len(all_dates)} 个交易日")

        for date in all_dates:
            self.today_bought = set()

            signals = {
                code: signal_table.get((code, date), 0.5)
                for code in self.stock_data
            }

            total_value = self._portfolio_value(date)

            # 单股止损（T+1保护）
            for code in list(self.holdings.keys()):
                if code in self.today_bought:
                    continue
                price = self._get_close(code, date)
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
                    price = self._get_close(code, date)
                    if price > 0:
                        self._sell(code, price, date, reason="signal")

            # 买入信号（按概率降序，优先买最强信号）
            for code, prob in sorted(signals.items(), key=lambda x: -x[1]):
                if len(self.holdings) >= MAX_HOLDINGS:
                    break
                if code in self.holdings:
                    continue
                if prob > BUY_THRESHOLD:
                    price = self._get_close(code, date)
                    if price > 0:
                        self._buy(code, price, date, total_value)

            self.nav_curve.append((date, self._portfolio_value(date)))

        nav_df = pd.DataFrame(self.nav_curve, columns=["date", "nav"])

        # 准备基准净值（沪深300）
        bm_nav = None
        if self.benchmark_df is not None and not self.benchmark_df.empty:
            bm = self.benchmark_df.set_index("date")["close"].reindex(nav_df["date"])
            bm = bm.ffill().bfill()
            bm_nav = (bm / bm.iloc[0]) * INIT_CAPITAL

        metrics = compute_metrics(nav_df, self.trades, benchmark_nav=bm_nav)
        self._save_report(nav_df, metrics, bm_nav)
        return metrics

    def _save_report(self, nav_df: pd.DataFrame, metrics: dict, bm_nav: pd.Series = None):
        """保存净值曲线图和绩效报告。"""
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(nav_df["date"], nav_df["nav"] / INIT_CAPITAL,
                linewidth=1.5, label="AlphaQuant", color="#2196F3")
        if bm_nav is not None:
            ax.plot(nav_df["date"], bm_nav / INIT_CAPITAL,
                    linewidth=1.2, label="沪深300", color="#FF9800", linestyle="--", alpha=0.8)
        ax.axhline(1.0, color="gray", linestyle=":", alpha=0.5)
        ax.set_title("AlphaQuant 回测净值曲线", fontsize=14)
        ax.set_ylabel("净值")
        ax.set_xlabel("日期")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()

        path = os.path.join(REPORTS_DIR, "backtest_nav.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info(f"报告已保存: {path}")

        print("\n" + "=" * 52)
        print("  AlphaQuant 回测绩效报告")
        print("=" * 52)
        for k, v in metrics.items():
            print(f"  {k:<16} {v}")
        print("=" * 52)
