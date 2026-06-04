# 交易执行：每日收盘后拉取行情、推理信号、执行买卖、更新账户

import os
import logging
from datetime import datetime

import torch
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    STOCK_POOL, BUY_THRESHOLD, SELL_THRESHOLD,
    MAX_HOLDINGS, MAX_POSITION_RATIO, MODEL_SAVE_DIR,
)
from data.realtime import fetch_all_realtime, update_all_caches, is_trade_day
from data.loader import load_stock_data
from features.builder import build_inference_sequence, load_scaler
from models.lstm_model import load_model
from paper_trading.account import VirtualAccount
from paper_trading.risk import RiskManager
from paper_trading.logger import TradeLogger

logger = logging.getLogger(__name__)


def _infer_prob(model, scaler, code: str) -> float:
    """推理单只股票的买入概率，失败返回 0.5。"""
    try:
        df = load_stock_data(code)
        if df is None or len(df) < 21:
            return 0.5
        X   = build_inference_sequence(df, scaler)
        X_t = torch.tensor(X, dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            return float(model(X_t).item())
    except Exception as e:
        logger.warning(f"[{code}] 推理失败: {e}")
        return 0.5


def run_daily_execution():
    """
    每日收盘后执行的完整流程：
    1. 更新行情缓存
    2. 模型推理
    3. 风控检查
    4. 执行交易
    5. 记录日志
    """
    if not is_trade_day():
        logger.info("今日非交易日，跳过执行")
        return

    logger.info(f"===== 每日执行开始 {datetime.now().strftime('%Y-%m-%d %H:%M')} =====")

    model_path = os.path.join(MODEL_SAVE_DIR, "lstm_best.pt")
    if not os.path.exists(model_path):
        logger.error("模型未找到，请先运行 python main.py --mode train")
        return

    model  = load_model(model_path)
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

    # 推理信号
    signals = {code: _infer_prob(model, scaler, code) for code in STOCK_POOL}

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
    手动触发信号计算（用于 --mode signal / 看板）。

    Returns:
        {stock_code: probability}
    """
    model_path = os.path.join(MODEL_SAVE_DIR, "lstm_best.pt")
    if not os.path.exists(model_path):
        logger.error("模型未找到，请先训练")
        return {}

    model  = load_model(model_path)
    scaler = load_scaler()

    return {code: _infer_prob(model, scaler, code) for code in STOCK_POOL}
