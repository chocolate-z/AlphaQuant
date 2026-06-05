# 交易执行：每日收盘后拉取行情、推理信号、执行买卖、更新账户

import os
import json
import logging
from datetime import datetime

import torch

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    STOCK_POOL, BUY_THRESHOLD, SELL_THRESHOLD,
    MAX_HOLDINGS, MAX_POSITION_RATIO, MODEL_SAVE_DIR,
    SIGNAL_CACHE_FILE,
)
from data.realtime import fetch_all_realtime, update_all_caches, is_trade_day
from data.loader import load_all_stocks, load_stock_data
from features.builder import build_inference_sequence, load_scaler, CROSS_FEATURES, compute_cross_sectional
from models.lstm_model import load_model, load_best_available
from models.trainer import predict_proba
from paper_trading.account import VirtualAccount
from paper_trading.risk import RiskManager
from paper_trading.logger import TradeLogger

logger = logging.getLogger(__name__)


def _compute_signals_batch(model, scaler, stock_data: dict = None) -> dict:
    """
    批量推理股票信号（一次性加载所有数据，避免重复 IO）。
    model 可以是单模型或集成模型列表，predict_proba 自动处理。
    stock_data 不传时回退 load_all_stocks()；传入时对指定股票池打分（与回测同口径）。

    Returns:
        {stock_code: probability}
    """
    if stock_data is None:
        stock_data = load_all_stocks()
    # 在整个池上算横截面排名（开启时，与训练/回测同口径）
    cs_map = compute_cross_sectional(stock_data) if CROSS_FEATURES else {}
    signals = {}
    for code, df in stock_data.items():
        if df is None or len(df) < 21:
            signals[code] = 0.5
            continue
        try:
            X = build_inference_sequence(df, scaler, cs_df=cs_map.get(code))
            signals[code] = float(predict_proba(model, X)[0])
        except Exception as e:
            logger.warning(f"[{code}] 推理失败: {e}")
            signals[code] = 0.5
    return signals


def _save_signal_cache(signals: dict):
    """将今日信号写入缓存文件，供 dashboard 读取。"""
    data = {
        "date":    datetime.today().strftime("%Y-%m-%d"),
        "time":    datetime.now().strftime("%H:%M:%S"),
        "signals": signals,
    }
    with open(SIGNAL_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def _cooldown_path():
    from config import LOGS_DIR
    return os.path.join(LOGS_DIR, "cooldown_state.json")


def _load_cooldown() -> dict:
    p = _cooldown_path()
    if os.path.exists(p):
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_cooldown(cd: dict):
    try:
        with open(_cooldown_path(), "w", encoding="utf-8") as f:
            json.dump(cd, f, ensure_ascii=False)
    except Exception:
        pass


def run_daily_execution():
    """
    每日收盘后执行：与回测引擎**完全一致**的策略——相对排名选股 + 大盘择时闸 +
    动态可交易池(按昨日流动性) + 最短持有/卖出冷却 + 止损止盈 + 组合止损。
    「信号扫全池、下单只在流动性好的票」。复用回测引擎已验证的择时/流动性判定，确保实盘=回测。

    时点说明：本流程于收盘后运行，用截至最新收盘的数据出信号，所持仓位自下一交易日生效
    （纸面记账近似），与回测的"信号滞后1日"口径一致，不引入未来函数。
    """
    from config import (DATA_CACHE_DIR, MAX_TRAIN_STOCKS, MAX_HOLDINGS, MAX_POSITION_RATIO,
                        STOP_LOSS_RATIO, TAKE_PROFIT_RATIO, RANK_SELL_BOTTOM,
                        MIN_HOLD_DAYS, COOLDOWN_DAYS, USE_MARKET_FILTER)
    from datetime import datetime as _dt
    import pandas as pd
    from data.loader import load_cached_stocks
    from backtest.engine import BacktestEngine

    if not is_trade_day():
        logger.info("今日非交易日，跳过执行")
        return
    logger.info(f"===== 每日执行开始 {datetime.now().strftime('%Y-%m-%d %H:%M')} =====")

    if not os.path.exists(os.path.join(MODEL_SAVE_DIR, "lstm_best.pt")):
        logger.error("模型未找到，请先训练（python main.py 选项1/f）")
        return
    model  = load_best_available()
    scaler = load_scaler()

    account      = VirtualAccount()
    account.new_day()
    risk         = RiskManager(account)
    trade_logger = TradeLogger()

    # ── 股票池 = 缓存的 sh/sz 主板（与回测/训练同口径）──
    update_all_caches()
    stock_data = load_cached_stocks(boards=("sh", "sz"), limit=MAX_TRAIN_STOCKS)
    if not stock_data:
        logger.warning("无缓存股票数据，跳过")
        return

    # 实时价（取不到则回退到缓存最新收盘）
    rt = {c: q["close"] for c, q in fetch_all_realtime(list(stock_data)).items() if q.get("close", 0) > 0}
    def price_of(c):
        if c in rt:
            return rt[c]
        df = stock_data.get(c)
        return float(df["close"].iloc[-1]) if df is not None and len(df) else 0.0

    account.update_prices({c: price_of(c) for c in account.holdings})

    # 组合止损 / 暂停
    if risk.check_portfolio_stop():
        logger.warning("组合止损触发，全部清仓")
        for c in list(account.holdings):
            if c in account.today_bought:
                continue
            p = price_of(c)
            if p > 0:
                sh = account.holdings[c]["shares"]
                account.sell(c, p)
                trade_logger.log_trade(c, "sell", p, sh, reason="portfolio_stop")
        trade_logger.log_daily_snapshot(account)
        return
    if risk.is_suspended():
        logger.info("模拟盘暂停中，跳过今日交易")
        trade_logger.log_daily_snapshot(account)
        return

    # ── 全池打分 ──
    logger.info("正在批量推理信号（全池打分）...")
    signals = _compute_signals_batch(model, scaler, stock_data=stock_data)
    _save_signal_cache(signals)
    logger.info(f"信号已缓存，共 {len(signals)} 只")

    # ── 复用回测引擎已验证的「大盘择时闸」「动态流动性」判定，保证实盘=回测 ──
    bench = {}
    mfp = os.path.join(DATA_CACHE_DIR, "market_features.csv")
    if os.path.exists(mfp):
        m = pd.read_csv(mfp, parse_dates=["date"])
        if "mkt_close" in m.columns:
            bench["沪深300"] = m[["date", "mkt_close"]].rename(columns={"mkt_close": "close"})
    eng = BacktestEngine(stock_data, model, benchmarks=bench)
    eng._build_market_trend()
    eng._build_liquidity()
    all_dates = sorted({d for df in stock_data.values() for d in df["date"]})
    ref = all_dates[-1] if all_dates else None
    bullish = eng._is_market_bullish(ref) if ref is not None else True

    today_str = _dt.today().strftime("%Y-%m-%d")
    def held_days(c):
        bd = account.holdings.get(c, {}).get("buy_date")
        if not bd:
            return 9999
        try:
            return (_dt.today() - _dt.strptime(bd, "%Y-%m-%d")).days
        except Exception:
            return 9999
    cd = _load_cooldown()
    def in_cooldown(c):
        sd = cd.get(c)
        if not sd:
            return False
        try:
            return (_dt.today() - _dt.strptime(sd, "%Y-%m-%d")).days < COOLDOWN_DAYS * 1.5
        except Exception:
            return False

    def do_sell(c, reason):
        if c in account.today_bought:
            return
        p = price_of(c)
        if p <= 0:
            return
        sh = account.holdings[c]["shares"]
        if account.sell(c, p):
            trade_logger.log_trade(c, "sell", p, sh, reason=reason)
            cd[c] = today_str

    # ── 决策：完全镜像回测引擎 ──
    if USE_MARKET_FILTER and not bullish:
        logger.info("大盘择时闸：趋势转空 → 清仓避熊，今日不开新仓")
        for c in list(account.holdings):
            do_sell(c, "market_bear")
    else:
        # 止损 / 止盈（风控，随时可触发）
        for c in list(account.holdings):
            if c in account.today_bought:
                continue
            p, cost = price_of(c), account.holdings[c]["cost"]
            if p <= 0 or cost <= 0:
                continue
            pnl = (p - cost) / cost
            if pnl < STOP_LOSS_RATIO:
                do_sell(c, "stop_loss")
            elif pnl >= TAKE_PROFIT_RATIO:
                do_sell(c, "take_profit")
        # 排名卖出：信号排名处于底部 + 已满足最短持有期
        asc = sorted(signals, key=lambda c: signals[c])
        n = max(len(asc), 1)
        for i, c in enumerate(asc):
            if (c in account.holdings and c not in account.today_bought
                    and held_days(c) >= MIN_HOLD_DAYS * 1.5 and (i / n) < RANK_SELL_BOTTOM):
                do_sell(c, "rank_signal")
        # 排名买入：信号最高、流动性达标、未持有、非冷却
        for c in sorted(signals, key=lambda c: -signals[c]):
            if len(account.holdings) >= MAX_HOLDINGS:
                break
            if c in account.holdings or in_cooldown(c) or not eng._is_liquid(c, ref):
                continue
            p = price_of(c)
            if p <= 0:
                continue
            shares = int(account.total_assets * MAX_POSITION_RATIO / p / 100) * 100
            if shares > 0 and account.buy(c, p, shares):
                trade_logger.log_trade(c, "buy", p, shares, reason=f"rank_{signals[c]:.2f}")

    _save_cooldown(cd)
    account.update_prices({c: price_of(c) for c in account.holdings})
    risk.update_peak()
    trade_logger.log_daily_snapshot(account)
    logger.info(f"===== 执行完成，总资产: ¥{account.total_assets:,.0f} =====")


def get_current_signals() -> dict:
    """
    获取当日信号：优先读缓存，缓存不存在则实时计算。

    Returns:
        {stock_code: probability}
    """
    # 优先读当日缓存
    if os.path.exists(SIGNAL_CACHE_FILE):
        try:
            with open(SIGNAL_CACHE_FILE, encoding="utf-8") as f:
                cached = json.load(f)
            cache_date = cached.get("date", "")
            if cache_date == datetime.today().strftime("%Y-%m-%d"):
                logger.info(f"使用信号缓存（{cached.get('time', '')}）")
                return cached.get("signals", {})
        except Exception:
            pass

    # 缓存无效则实时计算
    model_path = os.path.join(MODEL_SAVE_DIR, "lstm_best.pt")
    if not os.path.exists(model_path):
        logger.error("模型未找到，请先训练")
        return {}

    model   = load_best_available()
    scaler  = load_scaler()
    signals = _compute_signals_batch(model, scaler)
    _save_signal_cache(signals)
    return signals
