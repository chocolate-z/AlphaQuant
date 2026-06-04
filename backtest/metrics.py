# 绩效指标计算：年化收益、夏普、Calmar、Sortino、最大回撤、胜率

import numpy as np
import pandas as pd
from collections import defaultdict, deque
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RISK_FREE_RATE, INIT_CAPITAL


def compute_metrics(nav_df: pd.DataFrame, trades: list,
                    benchmark_nav: pd.Series = None) -> dict:
    """
    计算回测绩效指标。

    Args:
        nav_df: DataFrame with columns ['date', 'nav']
        trades: list of trade dicts
        benchmark_nav: 可选的基准净值序列（如沪深300），用于计算超额收益

    Returns:
        指标字典
    """
    if nav_df.empty:
        return {}

    navs    = nav_df["nav"].values
    returns = np.diff(navs) / (navs[:-1] + 1e-9)

    total_days    = (nav_df["date"].iloc[-1] - nav_df["date"].iloc[0]).days
    total_return  = (navs[-1] / INIT_CAPITAL) - 1
    annual_return = (1 + total_return) ** (365 / max(total_days, 1)) - 1

    # ── 夏普比率 ──────────────────────────────────────
    daily_rf = RISK_FREE_RATE / 252
    if len(returns) > 1 and returns.std() > 0:
        excess = returns - daily_rf
        sharpe = np.sqrt(252) * excess.mean() / (excess.std() + 1e-9)
    else:
        sharpe = 0.0

    # ── Sortino 比率（只惩罚下行波动）────────────────
    downside = returns[returns < daily_rf] - daily_rf
    if len(downside) > 1 and downside.std() > 0:
        sortino = np.sqrt(252) * (returns.mean() - daily_rf) / (downside.std() + 1e-9)
    else:
        sortino = 0.0

    # ── 最大回撤 ─────────────────────────────────────
    peak     = np.maximum.accumulate(navs)
    drawdown = (navs - peak) / (peak + 1e-9)
    max_dd   = drawdown.min()

    # ── Calmar 比率（年化收益/最大回撤绝对值）────────
    calmar = annual_return / (abs(max_dd) + 1e-9)

    # ── 胜率（FIFO 配对，正确处理同一股票多次交易）──
    buy_queues   = defaultdict(deque)
    wins         = 0
    total_sells  = 0

    for t in trades:
        if t.get("action") == "buy":
            buy_queues[t["code"]].append(float(t["price"]))
        elif t.get("action") == "sell":
            total_sells += 1
            q = buy_queues.get(t["code"])
            if q:
                buy_price = q.popleft()
                if float(t["price"]) > buy_price:
                    wins += 1

    win_rate = wins / total_sells if total_sells > 0 else 0.0

    # ── 超额收益（相对基准）──────────────────────────
    alpha_str = "N/A（未提供基准）"
    if benchmark_nav is not None and len(benchmark_nav) > 1:
        bm_return = (benchmark_nav.iloc[-1] / benchmark_nav.iloc[0]) - 1
        bm_annual = (1 + bm_return) ** (365 / max(total_days, 1)) - 1
        alpha_str = f"{(annual_return - bm_annual)*100:.2f}%"

    return {
        "总收益率":   f"{total_return*100:.2f}%",
        "年化收益率": f"{annual_return*100:.2f}%",
        "夏普比率":   f"{sharpe:.3f}",
        "Sortino比率": f"{sortino:.3f}",
        "Calmar比率":  f"{calmar:.3f}",
        "最大回撤":   f"{max_dd*100:.2f}%",
        "胜率":       f"{win_rate*100:.1f}%",
        "交易次数":   total_sells,
        "超额收益":   alpha_str,
        "最终资产":   f"¥{navs[-1]:,.0f}",
    }
