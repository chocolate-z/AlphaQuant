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
    RELATIVE_RANK_MODE, TOP_N_BUY, RANK_SELL_BOTTOM,
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
                 benchmarks: dict = None):
        """
        Args:
            benchmarks: {"上证指数": DataFrame(date,close), "沪深300": ..., ...}
        """
        self.stock_data = stock_data
        self.model      = model
        self.scaler     = scaler
        self.benchmarks = benchmarks or {}   # 多基准字典
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

            # ── 卖出判断 ────────────────────────────────
            # 单股止损（T+1保护，两种模式均执行）
            for code in list(self.holdings.keys()):
                if code in self.today_bought:
                    continue
                price = self._get_close(code, date)
                if price <= 0:
                    continue
                pnl = (price - self.holdings[code]["cost"]) / self.holdings[code]["cost"]
                if pnl < STOP_LOSS_RATIO:
                    self._sell(code, price, date, reason="stop_loss")

            if RELATIVE_RANK_MODE:
                # 相对排名模式：持仓中排名垫底（低于 RANK_SELL_BOTTOM）的卖出
                sorted_codes = sorted(signals.keys(), key=lambda c: signals[c])
                n = len(sorted_codes)
                for i, code in enumerate(sorted_codes):
                    if code not in self.holdings or code in self.today_bought:
                        continue
                    # 该股票在当日排名处于后 RANK_SELL_BOTTOM 分位
                    if (i / max(n, 1)) < RANK_SELL_BOTTOM:
                        price = self._get_close(code, date)
                        if price > 0:
                            self._sell(code, price, date, reason="rank_signal")
            else:
                # 绝对阈值模式
                for code in list(self.holdings.keys()):
                    if code in self.today_bought:
                        continue
                    if signals.get(code, 0.5) < SELL_THRESHOLD:
                        price = self._get_close(code, date)
                        if price > 0:
                            self._sell(code, price, date, reason="signal")

            # ── 买入判断 ────────────────────────────────
            ranked = sorted(signals.items(), key=lambda x: -x[1])

            if RELATIVE_RANK_MODE:
                # 相对排名模式：每天买概率最高的前 TOP_N_BUY 只
                candidates = [
                    (code, prob) for code, prob in ranked
                    if code not in self.holdings and code not in self.today_bought
                ][:TOP_N_BUY]
                for code, prob in candidates:
                    if len(self.holdings) >= MAX_HOLDINGS:
                        break
                    price = self._get_close(code, date)
                    if price > 0:
                        self._buy(code, price, date, total_value)
            else:
                # 绝对阈值模式
                for code, prob in ranked:
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

        # 将各基准指数对齐到回测日期，统一归一化为净值曲线
        bm_navs = {}
        for name, df in self.benchmarks.items():
            if df.empty:
                continue
            s = df.set_index("date")["close"].reindex(nav_df["date"]).ffill().bfill()
            if s.notna().any() and s.iloc[0] > 0:
                bm_navs[name] = (s / s.iloc[0]) * INIT_CAPITAL

        # 取沪深300（如有）作为夏普/超额收益计算基准
        hs300_nav = bm_navs.get("沪深300")

        metrics = compute_metrics(nav_df, self.trades, benchmark_nav=hs300_nav)
        self._save_report(nav_df, metrics, bm_navs)
        self._save_trade_log()
        return metrics

    def _save_trade_log(self):
        """打印交易明细到终端，并保存 CSV 到 reports/trade_log.csv。"""
        if not self.trades:
            print("\n  回测期间无任何交易记录\n")
            return

        df = pd.DataFrame(self.trades)
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df["action_cn"] = df["action"].map({"buy": "买入", "sell": "卖出"})
        df["commission"] = df["commission"].round(2)

        # 计算每笔卖出盈亏（匹配最近一次买入）
        pnl_map = {}
        for _, row in df.iterrows():
            code = row["code"]
            if row["action"] == "buy":
                pnl_map[code] = row["price"]
            elif row["action"] == "sell" and code in pnl_map:
                buy_p = pnl_map.pop(code)
                idx = df[(df["code"] == code) & (df["action"] == "sell") &
                         (df["date"] == row["date"])].index
                df.loc[idx, "pnl_pct"] = (row["price"] / buy_p - 1) * 100

        # 终端输出
        print("\n" + "=" * 72)
        print(f"  {'日期':<12} {'代码':<10} {'操作':<4} {'价格':>8} {'股数':>8} {'手续费':>8} {'涨跌%':>7} {'原因'}")
        print("-" * 72)
        for _, r in df.iterrows():
            reason = r.get("reason", "") or ""
            pnl    = r.get("pnl_pct", float("nan"))
            pnl_str = f"{pnl:+.2f}%" if not pd.isna(pnl) else "  —  "
            print(f"  {r['date']:<12} {r['code']:<10} {r['action_cn']:<4} "
                  f"{r['price']:>8.2f} {int(r['shares']):>8} {r['commission']:>8.2f} "
                  f"{pnl_str:>7} {reason}")
        print("=" * 72)
        print(f"  共 {len(df)} 笔交易（买入 {(df['action']=='buy').sum()} 笔，"
              f"卖出 {(df['action']=='sell').sum()} 笔）")
        print()

        # 保存 CSV
        csv_path = os.path.join(REPORTS_DIR, "trade_log.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"  交易明细已保存至: {csv_path}\n")

    def _save_report(self, nav_df: pd.DataFrame, metrics: dict, bm_navs: dict = None):
        """保存综合回测图表（净值曲线对比5指数 + 回撤 + 月度收益 + 交易盈亏分布）。"""
        bm_navs = bm_navs or {}

        fig = plt.figure(figsize=(16, 12))
        fig.suptitle("AlphaQuant 回测综合报告", fontsize=16, fontweight="bold")

        gs = fig.add_gridspec(3, 2, hspace=0.42, wspace=0.32)
        ax_nav  = fig.add_subplot(gs[0, :])   # 顶部跨全宽：净值曲线
        ax_dd   = fig.add_subplot(gs[1, 0])   # 中左：回撤曲线
        ax_mon  = fig.add_subplot(gs[1, 1])   # 中右：月度收益
        ax_pnl  = fig.add_subplot(gs[2, 0])   # 底左：盈亏分布
        ax_stat = fig.add_subplot(gs[2, 1])   # 底右：绩效指标文字

        nav_series = nav_df.set_index("date")["nav"]

        # ── 净值曲线（AlphaQuant + 5个基准指数）────────
        ax_nav.plot(nav_df["date"], nav_series / INIT_CAPITAL,
                    linewidth=2.2, label="AlphaQuant", color="#2196F3", zorder=5)

        # 各基准用不同颜色和线型
        bm_colors = {
            "上证指数": ("#FF5722", "--"),
            "深证成指": ("#9C27B0", "-."),
            "创业板指": ("#009688", ":"),
            "沪深300":  ("#FF9800", "--"),
            "上证50":   ("#795548", "-."),
        }
        for name, bm_nav in bm_navs.items():
            color, ls = bm_colors.get(name, ("#888888", "--"))
            ax_nav.plot(nav_df["date"], bm_nav / INIT_CAPITAL,
                        linewidth=1.2, label=name, color=color,
                        linestyle=ls, alpha=0.75)

        ax_nav.axhline(1.0, color="gray", linestyle=":", alpha=0.4)
        ax_nav.set_title("净值曲线对比（初始=1.0，蓝色=AlphaQuant）")
        ax_nav.set_ylabel("净值")
        ax_nav.legend(ncol=3, fontsize=9)
        ax_nav.grid(alpha=0.3)

        # ── 回撤曲线 ──────────────────────────────────
        roll_max = nav_series.cummax()
        drawdown = (nav_series - roll_max) / roll_max * 100
        ax_dd.fill_between(nav_df["date"], drawdown, 0,
                           color="#F44336", alpha=0.4, label="回撤%")
        ax_dd.plot(nav_df["date"], drawdown, color="#F44336", linewidth=0.8)
        ax_dd.set_title("回撤曲线（0%=历史最高点）")
        ax_dd.set_ylabel("回撤 (%)")
        ax_dd.grid(alpha=0.3)

        # ── 月度收益柱状图 ─────────────────────────────
        monthly = nav_series.resample("ME").last().pct_change().dropna() * 100
        if not monthly.empty:
            colors = ["#4CAF50" if v >= 0 else "#F44336" for v in monthly.values]
            ax_mon.bar(range(len(monthly)), monthly.values, color=colors, width=0.7)
            ax_mon.axhline(0, color="gray", linestyle="-", linewidth=0.8)
            ax_mon.set_xticks(range(len(monthly)))
            labels = [d.strftime("%y/%m") for d in monthly.index]
            ax_mon.set_xticklabels(labels, rotation=60, fontsize=7)
            ax_mon.set_title("月度收益（绿涨红跌）")
            ax_mon.set_ylabel("月收益 (%)")
            ax_mon.grid(alpha=0.3, axis="y")
        else:
            ax_mon.text(0.5, 0.5, "数据不足", ha="center", va="center",
                        transform=ax_mon.transAxes)
            ax_mon.set_title("月度收益")

        # ── 交易盈亏分布直方图 ──────────────────────────
        if self.trades:
            from collections import defaultdict, deque
            buy_q: dict = defaultdict(deque)
            pnls = []
            for t in self.trades:
                if t.get("action") == "buy":
                    buy_q[t["code"]].append(float(t["price"]))
                elif t.get("action") == "sell":
                    q = buy_q.get(t["code"])
                    if q:
                        bp = q.popleft()
                        pnl_pct = (float(t["price"]) - bp) / bp * 100
                        pnls.append(pnl_pct)
            if pnls:
                wins  = [p for p in pnls if p >= 0]
                loses = [p for p in pnls if p < 0]
                bins  = np.linspace(min(pnls) - 1, max(pnls) + 1, 30)
                ax_pnl.hist(loses, bins=bins, color="#F44336", alpha=0.7, label=f"亏损 {len(loses)}笔")
                ax_pnl.hist(wins,  bins=bins, color="#4CAF50", alpha=0.7, label=f"盈利 {len(wins)}笔")
                ax_pnl.axvline(0, color="gray", linestyle="--")
                ax_pnl.set_title(f"交易盈亏分布（共{len(pnls)}笔，胜率{len(wins)/len(pnls)*100:.1f}%）")
                ax_pnl.set_xlabel("单笔盈亏 (%)")
                ax_pnl.set_ylabel("笔数")
                ax_pnl.legend()
                ax_pnl.grid(alpha=0.3, axis="y")
            else:
                ax_pnl.text(0.5, 0.5, "无完整交易记录", ha="center", va="center",
                            transform=ax_pnl.transAxes)
                ax_pnl.set_title("交易盈亏分布")
        else:
            ax_pnl.text(0.5, 0.5, "无交易记录\n请检查模型训练质量", ha="center", va="center",
                        transform=ax_pnl.transAxes, fontsize=11, color="#F44336")
            ax_pnl.set_title("交易盈亏分布")

        # ── 绩效指标文字 ───────────────────────────────
        ax_stat.axis("off")
        lines = ["绩效指标汇总", "─" * 26]
        label_map = {
            "总收益率": "总收益率",
            "年化收益率": "年化收益率",
            "夏普比率": "夏普比率（>1佳）",
            "最大回撤": "最大回撤（越小越好）",
            "胜率": "胜率",
            "交易次数": "成交笔数",
            "Calmar比率": "Calmar比率",
        }
        for k, v in metrics.items():
            label = label_map.get(k, k)
            lines.append(f"{label:<18} {v}")
        text = "\n".join(lines)
        ax_stat.text(0.05, 0.95, text, transform=ax_stat.transAxes,
                     fontsize=10, verticalalignment="top", family="monospace",
                     bbox=dict(boxstyle="round", facecolor="#F5F5F5", alpha=0.8))
        ax_stat.set_title("绩效汇总")

        path = os.path.join(REPORTS_DIR, "backtest_report.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"回测报告已保存: {path}")
        print(f"\n  📊 回测图表已保存至: {path}")

        print("\n" + "=" * 54)
        print("  AlphaQuant 回测绩效报告")
        print("=" * 54)
        descs = {
            "总收益率":   "整个回测期间的总盈亏",
            "年化收益率":  "折算成每年的平均收益",
            "夏普比率":   "收益风险比（>1为佳）",
            "Sortino比率": "只考虑下跌风险的收益比",
            "Calmar比率":  "年化收益/最大回撤之比",
            "最大回撤":   "从最高点到最低点的最大跌幅",
            "胜率":       "盈利交易占总交易的比例",
            "交易次数":   "完整买卖交易的总笔数",
            "超额收益":   "相对沪深300基准的超额收益",
            "最终资产":   "回测结束时的账户总价值",
        }
        for k, v in metrics.items():
            desc = descs.get(k, "")
            print(f"  {k:<12} {str(v):<18}  {desc}")
        print("=" * 54)
