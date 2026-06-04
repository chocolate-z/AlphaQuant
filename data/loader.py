# 数据加载模块：搜狐前复权日K线 + 本地 CSV 缓存（无 AKShare 依赖）

import os
import re
import json
import time
import logging
from datetime import datetime, timedelta

import requests
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_CACHE_DIR, START_DATE, STOCK_POOL

logger = logging.getLogger(__name__)

_SINA_HEADERS = {
    "Referer": "http://finance.sina.com.cn/",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}
_SOHU_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}


def _cache_path(stock_code: str) -> str:
    return os.path.join(DATA_CACHE_DIR, f"{stock_code}.csv")


def _fetch_from_sohu(stock_code: str, start: str, end: str) -> pd.DataFrame:
    """
    从搜狐财经获取前复权日K线（境外可用）。
    URL: http://q.stock.sohu.com/hisHq?code=cn_{code}&start=YYYYMMDD&end=YYYYMMDD
         &stat=1&order=D&period=d&callback=historySearchHandler&rt=jsonp
    响应字段: [日期, 开盘, 收盘, 涨跌额, 涨跌幅%, 最低, 最高, 成交量(手), 成交额(万元), 换手率%]
    """
    pure_code = stock_code[2:]
    sohu_code = f"cn_{pure_code}"
    url = (
        f"http://q.stock.sohu.com/hisHq"
        f"?code={sohu_code}&start={start}&end={end}"
        f"&stat=1&order=D&period=d&callback=historySearchHandler&rt=jsonp"
    )

    try:
        resp = requests.get(url, timeout=20, headers=_SOHU_HEADERS, allow_redirects=True)
        resp.raise_for_status()
        text = resp.text

        m = re.search(r'historySearchHandler\((.*)\)', text, re.DOTALL)
        if not m:
            logger.warning(f"[{stock_code}] 搜狐响应解析失败（无 JSONP 包装）")
            return pd.DataFrame()

        data = json.loads(m.group(1))
        if not data or data[0].get("status") != 0:
            logger.warning(f"[{stock_code}] 搜狐返回 status 非 0")
            return pd.DataFrame()

        rows = data[0].get("hq", [])
        if not rows:
            return pd.DataFrame()

        records = []
        for row in rows:
            try:
                records.append({
                    "date":            pd.Timestamp(row[0]),
                    "open":            float(row[1]),
                    "close":           float(row[2]),
                    "high":            float(row[6]),
                    "low":             float(row[5]),
                    "volume":          float(row[7]) * 100,       # 手 → 股
                    "amount":          float(row[8]) * 10000,     # 万元 → 元
                    "pct_change":      float(str(row[4]).rstrip("%")),
                    "turnover":        float(str(row[9]).rstrip("%")) if row[9] else 0.0,
                    "volume_ratio":    1.0,
                    "main_net_inflow": 0.0,
                })
            except (ValueError, IndexError):
                continue

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df = df.sort_values("date").reset_index(drop=True)
        return df

    except Exception as e:
        logger.warning(f"[{stock_code}] 搜狐拉取失败: {e}")
        return pd.DataFrame()


def load_stock_data(stock_code: str, force_refresh: bool = False) -> pd.DataFrame:
    """
    加载单只股票历史数据，优先读取本地缓存。

    Args:
        stock_code: 股票代码，如 sh600519
        force_refresh: 强制重新从网络拉取

    Returns:
        包含 OHLCV 及衍生字段的 DataFrame
    """
    cache_file = _cache_path(stock_code)
    today_str = datetime.today().strftime("%Y%m%d")

    if not force_refresh and os.path.exists(cache_file):
        df = pd.read_csv(cache_file, parse_dates=["date"])
        df = df.sort_values("date").reset_index(drop=True)
        last_date = df["date"].iloc[-1]
        days_since = (datetime.today() - last_date).days

        if days_since <= 1:
            logger.info(f"[{stock_code}] 使用缓存（最新: {last_date.date()}）")
            return df

        # 增量更新
        incremental_start = (last_date + timedelta(days=1)).strftime("%Y%m%d")
        new_df = _fetch_from_sohu(stock_code, incremental_start, today_str)
        if not new_df.empty:
            df = pd.concat([df, new_df], ignore_index=True)
            df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
            df.to_csv(cache_file, index=False)
            logger.info(f"[{stock_code}] 增量更新 {len(new_df)} 条")
        return df

    # 全量拉取
    logger.info(f"[{stock_code}] 全量拉取数据...")
    df = _fetch_from_sohu(stock_code, START_DATE.replace("-", ""), today_str)
    if not df.empty:
        df.to_csv(cache_file, index=False)
        logger.info(f"[{stock_code}] 保存 {len(df)} 条到缓存")
    else:
        logger.warning(f"[{stock_code}] 数据为空，跳过")
    return df


def load_all_stocks(force_refresh: bool = False) -> dict:
    """
    批量加载股票池所有股票数据。

    Returns:
        {stock_code: DataFrame}
    """
    result = {}
    for code in STOCK_POOL:
        try:
            df = load_stock_data(code, force_refresh=force_refresh)
            if not df.empty:
                result[code] = df
                logger.info(f"[{code}] 加载 {len(df)} 条")
            else:
                logger.warning(f"[{code}] 数据为空，跳过")
            time.sleep(0.2)
        except Exception as e:
            logger.error(f"[{code}] 加载失败: {e}")
    return result


def get_stock_name(stock_code: str) -> str:
    """通过新浪行情接口获取股票名称（字段[0]）。"""
    try:
        url = f"https://hq.sinajs.cn/list={stock_code}"
        resp = requests.get(url, timeout=10, headers=_SINA_HEADERS)
        text = resp.content.decode("gb18030", errors="replace")
        m = re.search(r'"([^"]*)"', text)
        if m:
            fields = m.group(1).split(",")
            if fields and fields[0].strip():
                return fields[0].strip()
    except Exception:
        pass
    return stock_code
