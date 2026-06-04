# 实时/当日行情拉取：交易日收盘后自动追加到历史缓存

import os
import logging
from datetime import datetime, date

import akshare as ak
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_CACHE_DIR, STOCK_POOL
from data.loader import _cache_path

logger = logging.getLogger(__name__)


def is_trade_day(check_date: date = None) -> bool:
    """
    判断给定日期是否为交易日（排除周末和法定节假日）。

    Args:
        check_date: 待检查日期，默认今天

    Returns:
        True 表示是交易日
    """
    if check_date is None:
        check_date = date.today()

    if check_date.weekday() >= 5:
        return False

    try:
        trade_cal = ak.tool_trade_date_hist_sina()
        if trade_cal is not None and not trade_cal.empty:
            trade_dates = pd.to_datetime(trade_cal.iloc[:, 0]).dt.date.tolist()
            return check_date in trade_dates
    except Exception:
        pass

    # 降级：只排除周末
    return True


def fetch_realtime_quote(stock_code: str) -> dict:
    """
    获取单只股票当日实时行情快照。

    Args:
        stock_code: 如 sh600519

    Returns:
        包含 open/close/high/low/volume/amount/pct_change/turnover 的字典
    """
    pure_code = stock_code[2:]
    result = {
        "stock_code": stock_code,
        "date": datetime.today().date(),
        "open": 0.0, "close": 0.0, "high": 0.0, "low": 0.0,
        "volume": 0.0, "amount": 0.0, "pct_change": 0.0,
        "turnover": 0.0, "volume_ratio": 1.0, "main_net_inflow": 0.0,
    }

    try:
        df = ak.stock_zh_a_spot_em()
        if df is None or df.empty:
            return result

        col_map = {
            "代码": "code", "最新价": "close", "涨跌幅": "pct_change",
            "开盘价": "open", "最高价": "high", "最低价": "low",
            "成交量": "volume", "成交额": "amount", "换手率": "turnover",
            "量比": "volume_ratio",
        }
        df = df.rename(columns=col_map)
        row = df[df["code"] == pure_code]
        if row.empty:
            return result

        r = row.iloc[0]
        for field in ["open", "close", "high", "low", "volume", "amount", "pct_change", "turnover", "volume_ratio"]:
            if field in r.index:
                try:
                    result[field] = float(r[field])
                except (ValueError, TypeError):
                    pass

    except Exception as e:
        logger.error(f"[{stock_code}] 实时行情获取失败: {e}")

    return result


def fetch_all_realtime() -> dict:
    """批量获取股票池所有股票当日实时行情。"""
    results = {}
    for code in STOCK_POOL:
        q = fetch_realtime_quote(code)
        if q.get("close", 0) > 0:
            results[code] = q
    return results


def append_today_to_cache(stock_code: str, quote: dict) -> bool:
    """
    将今日行情追加到历史缓存 CSV。

    Args:
        stock_code: 股票代码
        quote: fetch_realtime_quote 返回的字典

    Returns:
        True 表示追加成功
    """
    cache_file = _cache_path(stock_code)
    if not os.path.exists(cache_file):
        logger.warning(f"[{stock_code}] 缓存文件不存在，无法追加")
        return False

    df = pd.read_csv(cache_file, parse_dates=["date"])
    today = pd.Timestamp(quote["date"])

    if today in df["date"].values:
        logger.info(f"[{stock_code}] 今日数据已存在，跳过追加")
        return True

    new_row = pd.DataFrame([{
        "date":             today,
        "open":             quote.get("open", 0),
        "close":            quote.get("close", 0),
        "high":             quote.get("high", 0),
        "low":              quote.get("low", 0),
        "volume":           quote.get("volume", 0),
        "amount":           quote.get("amount", 0),
        "pct_change":       quote.get("pct_change", 0),
        "turnover":         quote.get("turnover", 0),
        "volume_ratio":     quote.get("volume_ratio", 1.0),
        "main_net_inflow":  quote.get("main_net_inflow", 0.0),
    }])

    df = pd.concat([df, new_row], ignore_index=True)
    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(cache_file, index=False)
    logger.info(f"[{stock_code}] 今日数据追加成功")
    return True


def update_all_caches():
    """收盘后批量更新所有股票缓存。"""
    if not is_trade_day():
        logger.info("今日非交易日，跳过更新")
        return

    logger.info("开始更新当日行情缓存...")
    quotes = fetch_all_realtime()
    for code, quote in quotes.items():
        append_today_to_cache(code, quote)
    logger.info(f"缓存更新完成，共 {len(quotes)} 只")
