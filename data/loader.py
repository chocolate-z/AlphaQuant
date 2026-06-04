# 数据加载模块：通过 AKShare 获取日K线历史数据，本地 CSV 缓存

import os
import time
import logging
from datetime import datetime, timedelta

import akshare as ak
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_CACHE_DIR, START_DATE, STOCK_POOL

logger = logging.getLogger(__name__)


def _cache_path(stock_code: str) -> str:
    return os.path.join(DATA_CACHE_DIR, f"{stock_code}.csv")


def _fetch_from_akshare(stock_code: str, start: str, end: str) -> pd.DataFrame:
    """从 AKShare 获取前复权日K线，返回标准化 DataFrame。"""
    pure_code = stock_code[2:]

    try:
        df = ak.stock_zh_a_hist(
            symbol=pure_code,
            period="daily",
            start_date=start,
            end_date=end,
            adjust="qfq",
        )
    except Exception as e:
        logger.warning(f"[{stock_code}] AKShare 拉取失败: {e}")
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    col_map = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "涨跌幅": "pct_change",
        "换手率": "turnover",
    }
    df = df.rename(columns=col_map)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # 补充量比和主力净流入（默认值）
    if "volume_ratio" not in df.columns:
        df["volume_ratio"] = 1.0
    if "main_net_inflow" not in df.columns:
        df["main_net_inflow"] = 0.0

    df = _enrich_with_inflow(df, stock_code, pure_code)
    return df


def _enrich_with_inflow(df: pd.DataFrame, stock_code: str, pure_code: str) -> pd.DataFrame:
    """补充主力净流入字段（尽力而为，失败静默）。"""
    try:
        market = "sh" if stock_code.startswith("sh") else "sz"
        inflow_df = ak.stock_individual_fund_flow(stock=pure_code, market=market)
        if inflow_df is not None and not inflow_df.empty:
            col_map2 = {"日期": "date", "主力净流入净额": "main_net_inflow_new"}
            inflow_df = inflow_df.rename(columns=col_map2)
            inflow_df["date"] = pd.to_datetime(inflow_df["date"])
            if "main_net_inflow_new" in inflow_df.columns:
                df = df.merge(inflow_df[["date", "main_net_inflow_new"]], on="date", how="left")
                df["main_net_inflow"] = df["main_net_inflow_new"].fillna(0.0)
                df.drop(columns=["main_net_inflow_new"], inplace=True)
    except Exception:
        pass
    return df


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
        new_df = _fetch_from_akshare(stock_code, incremental_start, today_str)
        if not new_df.empty:
            df = pd.concat([df, new_df], ignore_index=True)
            df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
            df.to_csv(cache_file, index=False)
            logger.info(f"[{stock_code}] 增量更新 {len(new_df)} 条")
        return df

    # 全量拉取
    logger.info(f"[{stock_code}] 全量拉取数据...")
    df = _fetch_from_akshare(stock_code, START_DATE, today_str)
    if not df.empty:
        df.to_csv(cache_file, index=False)
        logger.info(f"[{stock_code}] 保存 {len(df)} 条到缓存")
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
            time.sleep(0.3)  # 避免频率限制
        except Exception as e:
            logger.error(f"[{code}] 加载失败: {e}")
    return result


def get_stock_name(stock_code: str) -> str:
    """获取股票名称（尽力而为）。"""
    try:
        pure_code = stock_code[2:]
        info = ak.stock_individual_info_em(symbol=pure_code)
        if info is not None and not info.empty:
            row = info[info["item"] == "股票简称"]
            if not row.empty:
                return str(row["value"].iloc[0])
    except Exception:
        pass
    return stock_code
