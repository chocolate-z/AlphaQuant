# 单股诊断：趋势判断、量能分析、AI概率、操作建议、风控价位

import os
import logging
import numpy as np
import pandas as pd
from scipy import stats

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_SAVE_DIR, STOP_LOSS_RATIO, LABEL_THRESHOLD
from data.loader import load_stock_data, get_stock_name
from features.builder import build_inference_sequence, load_scaler
from models.lstm_model import load_model

logger = logging.getLogger(__name__)


def _trend_analysis(df: pd.DataFrame) -> dict:
    """
    对最近20日收盘价做线性回归判断趋势。

    Returns:
        {trend, slope, r2, description}
    """
    closes = df["close"].tail(20).values
    if len(closes) < 5:
        return {"trend": "unknown", "slope": 0.0, "r2": 0.0, "description": "数据不足"}

    x = np.arange(len(closes))
    slope, _, r_val, _, _ = stats.linregress(x, closes)
    r2 = r_val ** 2

    if slope > 0 and r2 > 0.6:
        trend = "up"
        desc  = f"上升趋势（斜率 {slope:.3f}，R²={r2:.2f}）"
    elif slope < 0 and r2 > 0.6:
        trend = "down"
        desc  = f"下降趋势（斜率 {slope:.3f}，R²={r2:.2f}）"
    else:
        trend = "sideways"
        desc  = f"震荡趋势（R²={r2:.2f}，趋势不明显）"

    return {"trend": trend, "slope": float(slope), "r2": float(r2), "description": desc}


def _volume_analysis(df: pd.DataFrame) -> dict:
    """
    量能判断：今日量/20日均量 与价格涨跌方向。

    Returns:
        {state, ratio, description}
    """
    if len(df) < 21:
        return {"state": "unknown", "ratio": 1.0, "description": "数据不足"}

    vol_today = float(df["volume"].iloc[-1])
    vol_ma20  = float(df["volume"].tail(21).iloc[:-1].mean())
    ratio     = vol_today / max(vol_ma20, 1)
    pct       = float(df["pct_change"].iloc[-1]) if "pct_change" in df.columns else 0.0

    if ratio > 1.5 and pct > 0:
        state = "volume_up";   desc = f"放量上涨（量比 {ratio:.2f}x）"
    elif ratio > 1.5 and pct <= 0:
        state = "volume_down"; desc = f"放量下跌（量比 {ratio:.2f}x）"
    elif ratio < 0.7:
        state = "shrink";      desc = f"缩量（量比 {ratio:.2f}x）"
    else:
        state = "normal";      desc = f"量能正常（量比 {ratio:.2f}x）"

    return {"state": state, "ratio": float(ratio), "description": desc}


def _suggest_action(prob: float, trend: dict,
                    holdings_cost: float = None,
                    current_price: float = 0.0) -> dict:
    """
    综合 AI 概率 + 趋势生成操作建议及仓位建议。

    Returns:
        {action, reason, position_pct}
    """
    t = trend["trend"]

    # 持仓止盈止损优先
    if holdings_cost and current_price > 0 and holdings_cost > 0:
        pnl = (current_price - holdings_cost) / holdings_cost
        if pnl > LABEL_THRESHOLD:
            return {"action": "止盈卖出", "reason": f"浮盈 {pnl*100:.1f}% > {LABEL_THRESHOLD*100:.0f}%", "position_pct": 0}
        if pnl < STOP_LOSS_RATIO:
            return {"action": "止损卖出", "reason": f"浮亏 {pnl*100:.1f}% < {STOP_LOSS_RATIO*100:.0f}%", "position_pct": 0}

    if prob > 0.65:
        if t == "up":
            action, reason = "★ 建议买入", "AI高概率 + 上升趋势"
        else:
            action, reason = "◎ 观察买入", "AI高概率，但趋势不明"
    elif prob < 0.35 or t == "down":
        action, reason = "▼ 建议卖出/观望", "AI低概率或下降趋势"
    else:
        action, reason = "— 持仓观望", "概率中性，趋势震荡"

    # 建议仓位：概率越高上限越高，最高 20%
    position_pct = min(int(prob * 25), 20)

    return {"action": action, "reason": reason, "position_pct": position_pct}


def diagnose(stock_code: str, holdings_cost: float = None) -> dict:
    """
    对单只股票执行完整诊断。

    Args:
        stock_code: 如 sh600519
        holdings_cost: 持仓成本价（用于止盈止损判断），无持仓传 None

    Returns:
        诊断结果字典
    """
    logger.info(f"开始诊断: {stock_code}")

    df = load_stock_data(stock_code)
    if df is None or len(df) < 60:
        raise ValueError(f"[{stock_code}] 数据不足60条，无法诊断")

    latest        = df.iloc[-1]
    current_price = float(latest["close"])
    pct_change    = float(latest.get("pct_change", 0))
    turnover      = float(latest.get("turnover",   0))
    main_inflow   = float(latest.get("main_net_inflow", 0))

    # AI 概率
    ai_prob = 0.5
    try:
        model_path = os.path.join(MODEL_SAVE_DIR, "lstm_best.pt")
        model      = load_model(model_path)
        scaler     = load_scaler()
        X = build_inference_sequence(df, scaler)
        import torch
        X_t = torch.tensor(X, dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            ai_prob = float(model(X_t).item())
    except Exception as e:
        logger.warning(f"AI 推理失败（使用默认值 0.5）: {e}")

    trend  = _trend_analysis(df)
    volume = _volume_analysis(df)
    action = _suggest_action(ai_prob, trend, holdings_cost, current_price)

    stop_loss_price   = round(current_price * (1 + STOP_LOSS_RATIO), 2)
    take_profit_price = round(current_price * (1 + LABEL_THRESHOLD), 2)

    name = get_stock_name(stock_code)

    return {
        "stock_code":       stock_code,
        "stock_name":       name,
        "current_price":    current_price,
        "pct_change":       pct_change,
        "turnover":         turnover,
        "main_net_inflow":  main_inflow,
        "ai_prob":          ai_prob,
        "trend":            trend,
        "volume":           volume,
        "action":           action,
        "stop_loss_price":  stop_loss_price,
        "take_profit_price": take_profit_price,
        "holdings_cost":    holdings_cost,
    }
