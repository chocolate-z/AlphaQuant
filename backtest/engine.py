# 历史回测引擎：预计算信号、T+1、手续费、涨跌停、止损、止盈、基准对比

import os
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.viz import setup_chinese_font
setup_chinese_font()
from config import (
    INIT_CAPITAL, COMMISSION_BUY, COMMISSION_SELL,
    WINDOW_SIZE, BUY_THRESHOLD, SELL_THRESHOLD,
    STOP_LOSS_RATIO, TAKE_PROFIT_RATIO, MAX_POSITION_RATIO, MAX_HOLDINGS,
    RELATIVE_RANK_MODE, TOP_N_BUY, RANK_SELL_BOTTOM,
    MIN_HOLD_DAYS, COOLDOWN_DAYS, USE_MARKET_FILTER, MARKET_MA_DAYS,
    LIQ_WINDOW_DAYS, LIQ_MIN_AMOUNT_YI,
    REPORTS_DIR,
)
from models.lstm_model import LSTMModel
from models.trainer import predict_proba
from features.builder import build_sequences, FEATURE_NAMES, CROSS_FEATURES, compute_cross_sectional
from data.loader import get_stock_name

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
    - 支持止损 + 止盈
    - 支持沪深300基准对比
    """

    def __init__(self, stock_data: dict, model: LSTMModel, scaler=None,
                 benchmarks: dict = None, params: dict = None):
        """
        Args:
            benchmarks: {"上证指数": DataFrame(date,close), "沪深300": ..., ...}
            params:     可选的回测参数覆盖（网页面板用），未提供的项回退到 config 默认值。
                        支持键：init_capital, buy_threshold, sell_threshold, stop_loss,
                        take_profit, max_holdings, max_position, top_n_buy,
                        relative_rank, rank_sell_bottom
        """
        self.stock_data = stock_data
        self.model      = model
        self.scaler     = scaler
        self.benchmarks = benchmarks or {}

        # 可调参数：优先使用传入值，否则回退到 config 默认
        p = params or {}
        self.init_capital    = float(p.get("init_capital",    INIT_CAPITAL))
        self.buy_threshold   = float(p.get("buy_threshold",   BUY_THRESHOLD))
        self.sell_threshold  = float(p.get("sell_threshold",  SELL_THRESHOLD))
        self.stop_loss       = float(p.get("stop_loss",       STOP_LOSS_RATIO))
        self.take_profit     = float(p.get("take_profit",     TAKE_PROFIT_RATIO))
        self.max_holdings    = int(p.get("max_holdings",      MAX_HOLDINGS))
        self.max_position    = float(p.get("max_position",    MAX_POSITION_RATIO))
        self.top_n_buy       = int(p.get("top_n_buy",         TOP_N_BUY))
        self.relative_rank   = bool(p.get("relative_rank",    RELATIVE_RANK_MODE))
        self.rank_sell_bottom = float(p.get("rank_sell_bottom", RANK_SELL_BOTTOM))
        # 开始交易日：信号仍用完整历史预计算（特征 lookback 干净），但只从该日起
        # 建仓/记净值。用于「样本外回测」——模型训练截止日之后才开仓，结果才诚实。
        self.start_date = pd.Timestamp(p["start_date"]) if p.get("start_date") else None
        # 卖出冷却：刚卖出的股票在 cooldown_days 个交易日内不再买入，杜绝「当天/隔天
        # 卖了又买」的来回打脸式刷单（弱信号下排名抖动会导致这种交易，白送手续费）。
        self.cooldown_days = int(p.get("cooldown_days", COOLDOWN_DAYS))
        # 最短持有期：信号/排名类卖出至少持有 min_hold_days 个交易日才执行（止损/止盈
        # 不受限，可随时触发）。弱信号下排名每天抖动，最短持有期能大幅降低换手与手续费。
        self.min_hold_days = int(p.get("min_hold_days", MIN_HOLD_DAYS))
        # 大盘择时闸：沪深300 跌破均线则清仓避熊、不再开仓（只做多策略的救命阀）
        self.use_market_filter = bool(p.get("use_market_filter", USE_MARKET_FILTER))
        self.market_ma_days    = int(p.get("market_ma_days", MARKET_MA_DAYS))
        self._mkt_trend        = None   # {date: 是否多头}，run() 里构建
        # 动态可交易池（point-in-time）：买入只允许「截至当日近 N 日成交额中位数 ≥ 阈值」的票
        self.liq_window     = int(p.get("liq_window_days", LIQ_WINDOW_DAYS))
        self.liq_min_amount = float(p.get("liq_min_amount_yi", LIQ_MIN_AMOUNT_YI)) * 1e8
        self._liq           = None      # {code: {date: 近N日成交额中位数}}，run() 里构建

        self.cash         = float(self.init_capital)
        self.holdings     = {}
        self.today_bought = set()
        self.today_sold   = set()
        self.sold_on      = {}      # code -> 最近卖出日，用于冷却判断
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
        max_amount = total_value * self.max_position
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
        # 最短持有期：仅对信号/排名类卖出生效；止损/止盈属风控，任何时候都能卖
        if reason in ("signal", "rank_signal") and self.min_hold_days > 0:
            bd = self.holdings[code].get("buy_date")
            if bd is not None and (date - bd).days < self.min_hold_days * 1.5:
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
        self.today_sold.add(code)
        self.sold_on[code] = date

    def _in_cooldown(self, code: str, date: pd.Timestamp) -> bool:
        """刚卖出的股票是否还在冷却期内（冷却期内不再买入，防来回刷单）。"""
        if code in self.today_sold:
            return True
        if self.cooldown_days <= 0:
            return False
        last = self.sold_on.get(code)
        if last is None:
            return False
        # 自然日近似交易日：cooldown_days 个交易日 ≈ cooldown_days*1.5 自然日（含周末）
        return (date - last).days < self.cooldown_days * 1.5

    # ── 大盘择时闸 ────────────────────────────────────

    def _build_market_trend(self):
        """
        用沪深300构建每日多/空：收盘价 ≥ N日均线 视为多头（可持股），否则空头（清仓避熊）。
        优先用传入的「沪深300」基准，没有则读缓存 market_features.csv；都没有则关闭择时闸。
        """
        self._mkt_trend = None
        if not self.use_market_filter:
            return
        bm = self.benchmarks.get("沪深300") if self.benchmarks else None
        if bm is None or bm.empty:
            try:
                from config import DATA_CACHE_DIR
                fp = os.path.join(DATA_CACHE_DIR, "market_features.csv")
                if os.path.exists(fp):
                    mf = pd.read_csv(fp, parse_dates=["date"])
                    if "mkt_close" in mf.columns:
                        bm = mf[["date", "mkt_close"]].rename(columns={"mkt_close": "close"})
            except Exception:
                bm = None
        if bm is None or bm.empty:
            logger.warning("无沪深300数据，大盘择时闸已自动关闭")
            return
        bm = bm.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
        ma = bm["close"].rolling(self.market_ma_days, min_periods=1).mean()
        bull = (bm["close"].values >= ma.values)
        self._mkt_trend = {pd.Timestamp(d): bool(b) for d, b in zip(bm["date"], bull)}
        n_bull = int(sum(bull))
        logger.info(f"大盘择时闸已启用（沪深300 {self.market_ma_days}日均线）："
                    f"{n_bull}/{len(bull)} 个交易日为多头可持股")

    def _is_market_bullish(self, date: pd.Timestamp) -> bool:
        """该交易日大盘是否多头（可持股）。无数据/未启用 → 默认 True（不拦截）。"""
        if self._mkt_trend is None:
            return True
        v = self._mkt_trend.get(date)
        if v is not None:
            return v
        # 该日无精确记录：取之前最近一个交易日的趋势
        prior = [d for d in self._mkt_trend if d <= date]
        return self._mkt_trend[max(prior)] if prior else True

    # ── 动态可交易池（point-in-time 流动性）────────────

    def _build_liquidity(self):
        """
        逐股算「近 liq_window 个交易日成交额中位数」的滚动序列（只用当日及之前的数据，
        天然 point-in-time、无未来函数）。结果存 {code: {date: 中位成交额}}。
        """
        self._liq = {}
        if self.liq_min_amount <= 0:
            return
        for code, df in self.stock_data.items():
            if "amount" not in df.columns:
                continue
            amt = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
            med = amt.rolling(self.liq_window, min_periods=max(5, self.liq_window // 3)).median()
            self._liq[code] = dict(zip(df["date"], med.values))

    def _is_liquid(self, code: str, date: pd.Timestamp) -> bool:
        """截至 date，该票近 N 日成交额中位数是否达标（可买入）。无数据视为不可买（保守）。"""
        if self.liq_min_amount <= 0:
            return True
        m = (self._liq or {}).get(code, {}).get(date)
        if m is None or pd.isna(m):
            return False
        return m >= self.liq_min_amount

    # ── 信号预计算（核心优化）──────────────────────────

    def _precompute_signals(self) -> dict:
        """
        一次性预计算所有股票所有日期的信号。
        复杂度从 O(S*D*N) 降至 O(S*N)，速度提升约100x。

        Returns:
            {(code, date): probability}
        """
        signal_table = {}

        # 在整个回测股票池上算横截面排名（开启时，与训练同口径），再逐股推理
        cs_map = compute_cross_sectional(self.stock_data) if CROSS_FEATURES else {}

        for code, df in self.stock_data.items():
            if len(df) < WINDOW_SIZE + 5:
                continue
            try:
                X, _, _, dates = build_sequences(df, scaler=self.scaler, fit_scaler=False,
                                                 cs_df=cs_map.get(code))
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
            above_buy  = (p_arr > self.buy_threshold).sum()
            above_half = (p_arr > 0.5).sum()
            logger.info(
                f"概率分布 — min={p_arr.min():.3f}  mean={p_arr.mean():.3f}  "
                f"max={p_arr.max():.3f}  >0.5: {above_half}条  >{self.buy_threshold}: {above_buy}条"
            )
            if above_buy == 0:
                logger.warning(
                    f"⚠️  没有任何信号超过买入阈值 {self.buy_threshold}，将产生 0 笔交易。"
                    f"建议：① 重新训练模型（更多数据/更多轮次）"
                    f"② 或适当降低买入阈值（当前 {self.buy_threshold}）"
                )

        return signal_table

    # ── 主回测循环 ────────────────────────────────────

    def run(self) -> dict:
        """执行完整回测，返回绩效指标字典。"""
        from backtest.metrics import compute_metrics

        self._build_price_index()
        self._build_market_trend()      # 构建大盘择时闸
        self._build_liquidity()         # 构建 point-in-time 流动性（动态可交易池）
        logger.info("正在预计算所有信号（可能需要1-2分钟）...")
        signal_table = self._precompute_signals()

        all_dates_set = set()
        for df in self.stock_data.values():
            all_dates_set.update(df["date"].tolist())
        all_dates = sorted(all_dates_set)

        if not all_dates:
            print("\n  ⚠ 无有效股票数据，无法执行回测。请检查网络连接后重试。\n")
            return {}

        # 关键：建立「上一交易日」映射，用**昨日**的信号在**今日**成交，杜绝未来函数
        # （信号由截至某日收盘的数据算出，只能在它之后的交易日执行，不能当日算当日成交）。
        prev_day = {all_dates[i]: all_dates[i - 1] for i in range(1, len(all_dates))}

        # 样本外回测：信号已用完整历史预计算，这里把交易/净值起点推到 start_date 之后
        if self.start_date is not None:
            kept = [d for d in all_dates if d >= self.start_date]
            if kept:
                logger.info(f"样本外模式：仅从 {self.start_date.date()} 起开仓计净值"
                            f"（之前 {len(all_dates) - len(kept)} 个交易日仅用于特征 lookback）")
                all_dates = kept

        logger.info(f"回测区间: {all_dates[0].date()} ~ {all_dates[-1].date()}，共 {len(all_dates)} 个交易日")

        for date in all_dates:
            self.today_bought = set()
            self.today_sold   = set()

            # 用「上一交易日」的信号与大盘趋势做今日决策（无未来函数）；首日无前日则全 0.5（不动）
            sig_date = prev_day.get(date)
            signals = {
                code: (signal_table.get((code, sig_date), 0.5) if sig_date is not None else 0.5)
                for code in self.stock_data
            }

            total_value = self._portfolio_value(date)

            # ── 大盘择时闸：跌破均线 → 清仓避熊、本日不开新仓（用昨日趋势判定）──
            if self.use_market_filter and sig_date is not None and not self._is_market_bullish(sig_date):
                for code in list(self.holdings.keys()):
                    if code in self.today_bought:
                        continue
                    price = self._get_close(code, date)
                    if price > 0:
                        self._sell(code, price, date, reason="market_bear")
                self.nav_curve.append((date, self._portfolio_value(date)))
                continue

            # ── 卖出判断 ────────────────────────────────
            # 止损 + 止盈（T+1保护，两种模式均执行）
            for code in list(self.holdings.keys()):
                if code in self.today_bought:
                    continue
                price = self._get_close(code, date)
                if price <= 0:
                    continue
                pnl = (price - self.holdings[code]["cost"]) / self.holdings[code]["cost"]
                if pnl < self.stop_loss:
                    self._sell(code, price, date, reason="stop_loss")
                elif pnl >= self.take_profit:
                    self._sell(code, price, date, reason="take_profit")

            if self.relative_rank:
                sorted_codes = sorted(signals.keys(), key=lambda c: signals[c])
                n = len(sorted_codes)
                for i, code in enumerate(sorted_codes):
                    if code not in self.holdings or code in self.today_bought:
                        continue
                    if (i / max(n, 1)) < self.rank_sell_bottom:
                        price = self._get_close(code, date)
                        if price > 0:
                            self._sell(code, price, date, reason="rank_signal")
            else:
                for code in list(self.holdings.keys()):
                    if code in self.today_bought:
                        continue
                    if signals.get(code, 0.5) < self.sell_threshold:
                        price = self._get_close(code, date)
                        if price > 0:
                            self._sell(code, price, date, reason="signal")

            # ── 买入判断 ────────────────────────────────
            ranked = sorted(signals.items(), key=lambda x: -x[1])

            if self.relative_rank:
                candidates = [
                    (code, prob) for code, prob in ranked
                    if code not in self.holdings and code not in self.today_bought
                    and not self._in_cooldown(code, date)   # 刚卖出的不立刻买回，杜绝刷单
                    and self._is_liquid(code, sig_date)     # 按昨日流动性判定（动态池，无未来函数）
                ][:self.top_n_buy]
                for code, prob in candidates:
                    if len(self.holdings) >= self.max_holdings:
                        break
                    price = self._get_close(code, date)
                    if price > 0:
                        self._buy(code, price, date, total_value)
            else:
                for code, prob in ranked:
                    if len(self.holdings) >= self.max_holdings:
                        break
                    if code in self.holdings or self._in_cooldown(code, date):
                        continue
                    if not self._is_liquid(code, sig_date):   # 动态可交易池（按昨日流动性，无未来函数）
                        continue
                    if prob > self.buy_threshold:
                        price = self._get_close(code, date)
                        if price > 0:
                            self._buy(code, price, date, total_value)

            self.nav_curve.append((date, self._portfolio_value(date)))

        nav_df = pd.DataFrame(self.nav_curve, columns=["date", "nav"])

        bm_navs = {}
        for name, df in self.benchmarks.items():
            if df.empty:
                continue
            s = df.set_index("date")["close"].reindex(nav_df["date"]).ffill().bfill()
            if s.notna().any() and s.iloc[0] > 0:
                bm_navs[name] = (s / s.iloc[0]) * self.init_capital

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
        # 获取股票名称
        df["name"] = df["code"].apply(lambda c: get_stock_name(c)[:5])

        # 计算每笔卖出盈亏（匹配最近一次买入）。直接用当前行索引赋值，避免在 iterrows
        # 里再做 df[...] 过滤——那是 O(n²)，高换手回测下几千笔会卡住好几分钟。
        pnl_map = {}
        for i, row in df.iterrows():
            code = row["code"]
            if row["action"] == "buy":
                pnl_map[code] = row["price"]
            elif row["action"] == "sell" and code in pnl_map:
                buy_p = pnl_map.pop(code)
                df.loc[i, "pnl_pct"] = (row["price"] / buy_p - 1) * 100

        # 终端输出
        print("\n" + "=" * 80)
        print(f"  {'日期':<12} {'代码':<10} {'名称':<8} {'操作':<4} {'价格':>8} {'股数':>8} {'手续费':>8} {'涨跌%':>7} {'原因'}")
        print("-" * 80)
        for _, r in df.iterrows():
            reason  = r.get("reason", "") or ""
            pnl     = r.get("pnl_pct", float("nan"))
            pnl_str = f"{pnl:+.2f}%" if not pd.isna(pnl) else "  —  "
            name    = r.get("name", "")
            print(f"  {r['date']:<12} {r['code']:<10} {name:<8} {r['action_cn']:<4} "
                  f"{r['price']:>8.2f} {int(r['shares']):>8} {r['commission']:>8.2f} "
                  f"{pnl_str:>7} {reason}")
        print("=" * 80)
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
        ax_nav  = fig.add_subplot(gs[0, :])
        ax_dd   = fig.add_subplot(gs[1, 0])
        ax_mon  = fig.add_subplot(gs[1, 1])
        ax_pnl  = fig.add_subplot(gs[2, 0])
        ax_stat = fig.add_subplot(gs[2, 1])

        nav_series = nav_df.set_index("date")["nav"]

        # ── 净值曲线（AlphaQuant + 5个基准指数）────────
        ax_nav.plot(nav_df["date"], nav_series / self.init_capital,
                    linewidth=2.2, label="AlphaQuant", color="#2196F3", zorder=5)

        bm_colors = {
            "上证指数": ("#FF5722", "--"),
            "深证成指": ("#9C27B0", "-."),
            "创业板指": ("#009688", ":"),
            "沪深300":  ("#FF9800", "--"),
            "上证50":   ("#795548", "-."),
        }
        for name, bm_nav in bm_navs.items():
            color, ls = bm_colors.get(name, ("#888888", "--"))
            ax_nav.plot(nav_df["date"], bm_nav / self.init_capital,
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
            "总收益率":   "总收益率",
            "年化收益率": "年化收益率",
            "夏普比率":   "夏普比率（>1佳）",
            "最大回撤":   "最大回撤（越小越好）",
            "胜率":       "胜率",
            "交易次数":   "成交笔数",
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
