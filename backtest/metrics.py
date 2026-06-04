# 绩效指标计算：年化收益、夏普、最大回撤、胜率、超额收益

import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RISK_FREE_RATE, INIT_CAPITAL


def compute_metrics(nav_df: pd.DataFrame, trades: list) -> dict:
    """
    计算回测绩效指标。

    Args:
        nav_df: DataFrame with columns ['date', 'nav']
        trades: list of trade dicts

    Returns:
        指标字典
    """
    if nav_df.empty:
        return {}

    navs    = nav_df["nav"].values
    returns = np.diff(navs) / (navs[:-1] + 1e-9)

    total_days   = (nav_df["date"].iloc[-1] - nav_df["date"].iloc[0]).days
    total_return = (navs[-1] / INIT_CAPITAL) - 1
    annual_return = (1 + total_return) ** (365 / max(total_days, 1)) - 1

    if len(returns) > 1 and returns.std() > 0:
        daily_rf = RISK_FREE_RATE / 252
        excess   = returns - daily_rf
        sharpe   = np.sqrt(252) * excess.mean() / (excess.std() + 1e-9)
    else:
        sharpe = 0.0

    peak      = np.maximum.accumulate(navs)
    drawdown  = (navs - peak) / (peak + 1e-9)
    max_dd    = drawdown.min()

    # 胜率：卖出时价格高于对应买入价格的比例
    buy_map = {}
    for t in trades:
        if t.get("action") == "buy":
            buy_map[t["code"]] = t["price"]

    sell_trades = [t for t in trades if t.get("action") == "sell"]
    win = sum(1 for t in sell_trades if t["price"] > buy_map.get(t["code"], t["price"]))
    win_rate = win / len(sell_trades) if sell_trades else 0.0

    return {
        "总收益率":   f"{total_return*100:.2f}%",
        "年化收益率": f"{annual_return*100:.2f}%",
        "夏普比率":   f"{sharpe:.3f}",
        "最大回撤":   f"{max_dd*100:.2f}%",
        "胜率":       f"{win_rate*100:.1f}%",
        "交易次数":   len(sell_trades),
        "最终资产":   f"¥{navs[-1]:,.0f}",
    }
