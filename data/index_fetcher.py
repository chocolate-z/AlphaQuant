# 指数历史行情抓取：搜狐前复权日线（与 loader.py 同源，境外可达）

import re
import json
import time
import random
import logging
from datetime import datetime

import requests
import pandas as pd

logger = logging.getLogger(__name__)

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0",
]

def _headers() -> dict:
    return {
        "User-Agent": random.choice(_USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Referer": "https://q.stock.sohu.com/",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Cache-Control": "max-age=0",
    }

# 5个对比基准指数
BENCHMARKS = {
    "上证指数":  "cn_s_sh000001",
    "深证成指":  "cn_s_sz399001",
    "创业板指":  "cn_s_sz399006",
    "沪深300":   "cn_s_sh000300",
    "上证50":    "cn_s_sh000016",
}


def fetch_index_history(sohu_code: str, start: str, end: str) -> pd.DataFrame:
    """
    从搜狐财经拉取指数日线（前复权）。
    sohu_code 示例: cn_s_sh000001（指数代码需加 s_ 前缀）
    """
    url = (
        f"https://q.stock.sohu.com/hisHq"
        f"?code={sohu_code}&start={start}&end={end}"
        f"&stat=1&order=D&period=d&callback=historySearchHandler&rt=jsonp"
    )

    for attempt in range(4):
        try:
            resp = requests.get(url, timeout=20, headers=_headers(), allow_redirects=True)
            if resp.status_code in (503, 429):
                wait = 3 * (2 ** attempt) + random.uniform(0, 2)
                logger.warning(f"[{sohu_code}] 限流 {resp.status_code}，{wait:.1f}s 后重试")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            break
        except requests.exceptions.RequestException as e:
            if attempt == 3:
                logger.warning(f"[{sohu_code}] 指数拉取失败: {e}")
                return pd.DataFrame()
            time.sleep(2 ** attempt)
    else:
        return pd.DataFrame()

    try:
        m = re.search(r'historySearchHandler\((.*)\)', resp.text, re.DOTALL)
        if not m:
            return pd.DataFrame()
        data = json.loads(m.group(1))
        if not data or data[0].get("status") != 0:
            return pd.DataFrame()

        rows = data[0].get("hq", [])
        records = []
        for row in rows:
            try:
                records.append({
                    "date":  pd.Timestamp(row[0]),
                    "close": float(row[2]),
                })
            except (ValueError, IndexError):
                continue

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
        return df

    except Exception as e:
        logger.warning(f"[{sohu_code}] 指数解析失败: {e}")
        return pd.DataFrame()


def fetch_all_benchmarks(start: str, end: str) -> dict:
    """
    批量拉取所有基准指数历史数据。

    Args:
        start: YYYYMMDD
        end:   YYYYMMDD

    Returns:
        {"上证指数": DataFrame(date, close), ...}
    """
    result = {}
    for name, code in BENCHMARKS.items():
        df = fetch_index_history(code, start, end)
        if not df.empty:
            result[name] = df
            logger.info(f"[{name}] 指数加载 {len(df)} 条")
        else:
            logger.warning(f"[{name}] 指数数据为空，跳过")
        time.sleep(random.uniform(0.3, 0.8))  # 避免连续请求被限流
    return result
