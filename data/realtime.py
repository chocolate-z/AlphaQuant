# 实时/当日行情拉取：新浪 hq.sinajs.cn（GB18030，需 Referer）

import os
import re
import random
import logging
from datetime import datetime, date

import requests
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import STOCK_POOL
from data.loader import _cache_path, _USER_AGENTS

logger = logging.getLogger(__name__)

def _sina_headers() -> dict:
    return {
        "Referer": "http://finance.sina.com.cn/",
        "User-Agent": random.choice(_USER_AGENTS),
        "Accept-Language": "zh-CN,zh;q=0.9",
    }


def is_trade_day(check_date: date = None) -> bool:
    """
    判断给定日期是否为交易日（排除周末）。
    注：节假日接口 timor.tech 偶发 Cloudflare 挑战，降级为仅排除周末。
    """
    if check_date is None:
        check_date = date.today()
    return check_date.weekday() < 5


def _parse_sina_quote(code: str, text: str) -> dict:
    """
    解析新浪行情文本中单只股票的数据。
    A股字段索引：[0]名称 [1]今开 [2]昨收 [3]现价 [4]最高 [5]最低 [8]成交量(股) [9]成交额(元)
    """
    result = {
        "stock_code": code,
        "date": datetime.today().date(),
        "open": 0.0, "close": 0.0, "high": 0.0, "low": 0.0,
        "volume": 0.0, "amount": 0.0, "pct_change": 0.0,
        "turnover": 0.0, "volume_ratio": 1.0, "main_net_inflow": 0.0,
    }

    pattern = rf'hq_str_{re.escape(code)}="([^"]*)"'
    m = re.search(pattern, text)
    if not m:
        return result

    fields = m.group(1).split(",")
    if len(fields) < 10:
        return result

    try:
        open_  = float(fields[1])
        prev   = float(fields[2])
        close  = float(fields[3])
        high   = float(fields[4])
        low    = float(fields[5])
        volume = float(fields[8])
        amount = float(fields[9])

        if close == 0 and fields[6]:
            close = float(fields[6])  # 买一价兜底
        if close == 0:
            close = prev

        pct = (close - prev) / prev * 100 if prev > 0 else 0.0

        result.update({
            "open": open_, "close": close, "high": high, "low": low,
            "volume": volume, "amount": amount, "pct_change": round(pct, 2),
        })
    except (ValueError, ZeroDivisionError):
        pass

    return result


def fetch_realtime_quote(stock_code: str) -> dict:
    """
    获取单只股票当日实时行情（新浪 hq.sinajs.cn）。

    Args:
        stock_code: 如 sh600519

    Returns:
        包含 open/close/high/low/volume/amount/pct_change 的字典
    """
    result = {
        "stock_code": stock_code,
        "date": datetime.today().date(),
        "open": 0.0, "close": 0.0, "high": 0.0, "low": 0.0,
        "volume": 0.0, "amount": 0.0, "pct_change": 0.0,
        "turnover": 0.0, "volume_ratio": 1.0, "main_net_inflow": 0.0,
    }
    try:
        url = f"https://hq.sinajs.cn/list={stock_code}"
        resp = requests.get(url, timeout=10, headers=_sina_headers())
        text = resp.content.decode("gb18030", errors="replace")
        result = _parse_sina_quote(stock_code, text)
    except Exception as e:
        logger.error(f"[{stock_code}] 实时行情获取失败: {e}")
    return result


def fetch_all_realtime() -> dict:
    """
    批量获取股票池当日实时行情（单次请求多代码，每批 20 只）。
    """
    if not STOCK_POOL:
        return {}

    batch_size = 20
    results = {}

    for i in range(0, len(STOCK_POOL), batch_size):
        batch = STOCK_POOL[i: i + batch_size]
        codes_str = ",".join(batch)
        try:
            url = f"https://hq.sinajs.cn/list={codes_str}"
            resp = requests.get(url, timeout=15, headers=_sina_headers())
            text = resp.content.decode("gb18030", errors="replace")
            for code in batch:
                q = _parse_sina_quote(code, text)
                if q.get("close", 0) > 0:
                    results[code] = q
        except Exception as e:
            logger.error(f"批量行情请求失败 (batch {i//batch_size}): {e}")

    return results


def append_today_to_cache(stock_code: str, quote: dict) -> bool:
    """
    将今日行情追加到历史缓存 CSV。

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
