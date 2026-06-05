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
from features.builder import build_inference_sequence, load_scaler
from models.lstm_model import load_model, load_best_available
from models.trainer import predict_proba
from paper_trading.account import VirtualAccount
from paper_trading.risk import RiskManager
from paper_trading.logger import TradeLogger

logger = logging.getLogger(__name__)


def _compute_signals_batch(model, scaler) -> dict:
    """
    批量推理所有股票信号（一次性加载所有数据，避免重复 IO）。
    model 可以是单模型或集成模型列表，predict_proba 自动处理。

    Returns:
        {stock_code: probability}
    """
    stock_data = load_all_stocks()
    signals = {}
    for code, df in stock_data.items():
        if df is None or len(df) < 21:
            signals[code] = 0.5
            continue
        try:
            X = build_inference_sequence(df, scaler)
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


def run_daily_execution():
    """
    每日收盘后执行的完整流程：
    1. 更新行情缓存
    2. 批量模型推理（改进：一次加载所有数据）
    3. 保存信号缓存（供 dashboard 使用）
    4. 风控检查
    5. 执行交易
    6. 记录日志
    """
    if not is_trade_day():
        logger.info("今日非交易日，跳过执行")
        return

    logger.info(f"===== 每日执行开始 {datetime.now().strftime('%Y-%m-%d %H:%M')} =====")

    model_path = os.path.join(MODEL_SAVE_DIR, "lstm_best.pt")
    if not os.path.exists(model_path):
        logger.error("模型未找到，请先运行 python main.py --mode train")
        return

    model  = load_best_available()
    scaler = load_scaler()

    account      = VirtualAccount()
    account.new_day()
    risk         = RiskManager(account)
    trade_logger = TradeLogger()

    update_all_caches()
    quotes = fetch_all_realtime()

    if not quotes:
        logger.warning("未获取到任何行情，跳过")
        return

    prices = {code: q["close"] for code, q in quotes.items() if q.get("close", 0) > 0}
    account.update_prices(prices)

    # 组合止损检查
    if risk.check_portfolio_stop():
        logger.warning("组合止损触发，全部清仓")
        for code in list(account.holdings.keys()):
            price = prices.get(code, 0)
            if price > 0:
                shares = account.holdings[code]["shares"]
                account.sell(code, price)
                trade_logger.log_trade(code, "sell", price, shares, reason="portfolio_stop")
        trade_logger.log_daily_snapshot(account)
        return

    if risk.is_suspended():
        logger.info("模拟盘暂停中，跳过今日交易")
        trade_logger.log_daily_snapshot(account)
        return

    # 批量推理信号
    logger.info("正在批量推理信号...")
    signals = _compute_signals_batch(model, scaler)
    _save_signal_cache(signals)
    logger.info(f"信号已缓存，共 {len(signals)} 只")

    # 单股止损
    for code in list(account.holdings.keys()):
        if code in account.today_bought:
            continue
        price = prices.get(code, 0)
        if price > 0 and risk.check_single_stop_loss(code, price):
            shares = account.holdings[code]["shares"]
            account.sell(code, price)
            trade_logger.log_trade(code, "sell", price, shares, reason="stop_loss")

    # 卖出信号
    for code in list(account.holdings.keys()):
        if code in account.today_bought:
            continue
        if signals.get(code, 0.5) < SELL_THRESHOLD:
            price = prices.get(code, 0)
            if price > 0:
                shares = account.holdings[code]["shares"]
                account.sell(code, price)
                trade_logger.log_trade(code, "sell", price, shares,
                                       reason=f"signal_{signals[code]:.2f}")

    # 买入信号
    for code, prob in sorted(signals.items(), key=lambda x: -x[1]):
        if len(account.holdings) >= MAX_HOLDINGS:
            break
        if code in account.holdings:
            continue
        if prob > BUY_THRESHOLD:
            price = prices.get(code, 0)
            if price <= 0:
                continue
            shares = int(account.total_assets * MAX_POSITION_RATIO / price / 100) * 100
            if shares > 0:
                account.buy(code, price, shares)
                trade_logger.log_trade(code, "buy", price, shares,
                                       reason=f"signal_{prob:.2f}")

    account.update_prices(prices)
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
