# 数据加载：搜狐前复权日K线（主）+ 腾讯前复权K线（备）+ 本地CSV缓存

import os
import re
import json
import time
import random
import logging
from datetime import datetime, timedelta

import requests
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_CACHE_DIR, START_DATE, STOCK_POOL,
                    FULL_STOCK_COUNT, QUICK_STOCK_COUNT, QUICK_HISTORY_YEARS, STOCK_LIST_CACHE)

logger = logging.getLogger(__name__)

# ── 请求头 ────────────────────────────────────────────────────────────

# 轮换 User-Agent，降低被限流概率
_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
]

_SINA_HEADERS = {
    "Referer": "http://finance.sina.com.cn/",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}

def _sina_headers() -> dict:
    return {
        "Referer": "http://finance.sina.com.cn/",
        "User-Agent": random.choice(_USER_AGENTS),
        "Accept-Language": "zh-CN,zh;q=0.9",
    }

def _sohu_headers() -> dict:
    """模拟 Chrome/Edge 浏览器直接导航的完整请求头。"""
    return {
        "User-Agent": random.choice(_USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://q.stock.sohu.com/",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0",
    }

def _tencent_headers() -> dict:
    return {
        "User-Agent": random.choice(_USER_AGENTS),
        "Referer": "https://gu.qq.com/",
        "Accept-Language": "zh-CN,zh;q=0.9",
    }

def _throttle(base: float = 1.2):
    """随机延迟：base ± 50%，避免固定间隔被识别。"""
    time.sleep(base * (0.5 + random.random()))


# ── 搜狐数据源 ────────────────────────────────────────────────────────

def _fetch_from_sohu(stock_code: str, start: str, end: str) -> pd.DataFrame:
    """
    搜狐前复权日K线（主力源，境外可达）。
    字段: [日期, 开盘, 收盘, 涨跌额, 涨跌幅%, 最低, 最高, 成交量(手), 成交额(万元), 换手率%]
    """
    pure_code = stock_code[2:]
    url = (
        f"https://q.stock.sohu.com/hisHq"
        f"?code=cn_{pure_code}&start={start}&end={end}"
        f"&stat=1&order=D&period=d&callback=historySearchHandler&rt=jsonp"
    )

    for attempt in range(4):
        try:
            resp = requests.get(url, timeout=20, headers=_sohu_headers(), allow_redirects=True)
            if resp.status_code in (503, 429):
                wait = 3 * (2 ** attempt) + random.uniform(0, 2)
                logger.warning(f"[{stock_code}] 搜狐 {resp.status_code} 限流，{wait:.1f}s 后重试（第{attempt+1}/4次）")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            break
        except requests.exceptions.RequestException as e:
            if attempt == 3:
                logger.warning(f"[{stock_code}] 搜狐连接失败: {e}")
                return pd.DataFrame()
            time.sleep(3 * (2 ** attempt))
    else:
        return pd.DataFrame()

    try:
        m = re.search(r'historySearchHandler\((.*)\)', resp.text, re.DOTALL)
        if not m:
            return pd.DataFrame()
        data = json.loads(m.group(1))
        if not data or data[0].get("status") != 0:
            return pd.DataFrame()

        records = []
        for row in data[0].get("hq", []):
            try:
                records.append({
                    "date":      pd.Timestamp(row[0]),
                    "open":      float(row[1]),
                    "close":     float(row[2]),
                    "high":      float(row[6]),
                    "low":       float(row[5]),
                    "volume":    float(row[7]) * 100,
                    "amount":    float(row[8]) * 10000,
                    "pct_change": float(str(row[4]).rstrip("%")),
                    "turnover":  float(str(row[9]).rstrip("%")) if row[9] else 0.0,
                })
            except (ValueError, IndexError):
                continue

        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
        return df

    except Exception as e:
        logger.warning(f"[{stock_code}] 搜狐解析失败: {e}")
        return pd.DataFrame()


# ── 腾讯数据源（备用）────────────────────────────────────────────────

def _fetch_from_tencent(stock_code: str, start: str, end: str) -> pd.DataFrame:
    """
    腾讯前复权日K线（备用源，境外可达，已验证）。
    接口: web.ifzq.gtimg.cn/appstock/app/fqkline/get
    字段: [日期, 开盘, 收盘, 最高, 最低, 成交量(手)]
    注：腾讯不含成交额和换手率，填0即可（特征层不使用这两列）
    """
    start_fmt = f"{start[:4]}-{start[4:6]}-{start[6:]}"
    end_fmt   = f"{end[:4]}-{end[4:6]}-{end[6:]}"
    url = (
        f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
        f"?param={stock_code},day,{start_fmt},{end_fmt},3000,qfq"
    )

    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=20, headers=_tencent_headers())
            if resp.status_code in (503, 429):
                wait = 2 * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"[{stock_code}] 腾讯限流，{wait:.1f}s 后重试")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            break
        except requests.exceptions.RequestException as e:
            if attempt == 2:
                logger.warning(f"[{stock_code}] 腾讯连接失败: {e}")
                return pd.DataFrame()
            time.sleep(2 ** attempt)
    else:
        return pd.DataFrame()

    try:
        data = resp.json()
        if data.get("code") != 0:
            return pd.DataFrame()

        stock_data = data.get("data", {}).get(stock_code, {})
        # 优先取前复权数据 qfqday，没有则取 day
        rows = stock_data.get("qfqday") or stock_data.get("day") or []
        if not rows:
            return pd.DataFrame()

        records = []
        for row in rows:
            try:
                records.append({
                    "date":      pd.Timestamp(row[0]),
                    "open":      float(row[1]),
                    "close":     float(row[2]),
                    "high":      float(row[3]),
                    "low":       float(row[4]),
                    "volume":    float(row[5]) * 100,   # 手 → 股
                    "amount":    0.0,
                    "pct_change": 0.0,                  # 后续特征层用收益率计算
                    "turnover":  0.0,
                })
            except (ValueError, IndexError):
                continue

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
        # 腾讯 pct_change 从收盘价推算
        df["pct_change"] = df["close"].pct_change().fillna(0) * 100
        return df

    except Exception as e:
        logger.warning(f"[{stock_code}] 腾讯解析失败: {e}")
        return pd.DataFrame()


# ── 新浪数据源（第三备用）────────────────────────────────────────────

def _fetch_from_sina(stock_code: str, start: str, end: str) -> pd.DataFrame:
    """
    新浪财经日K线（第三备用源）。
    接口: money.finance.sina.com.cn/quotes_service/api/json_v2.php
    注意：单次最多返回 1023 条，需分批拉取覆盖完整历史。
    字段: d(日期) o(开) h(高) l(低) c(收) v(量/手) amount(额/元)
    非前复权，但特征层使用的全是相对量（收益率/比率），影响可接受。
    """
    start_dt = datetime.strptime(start, "%Y%m%d")
    end_dt   = datetime.strptime(end,   "%Y%m%d")

    all_records = []
    batch_days  = 1000            # 每批约1000个交易日（约4年）
    cur_end     = end_dt

    while cur_end > start_dt:
        cur_start = max(start_dt, cur_end - timedelta(days=batch_days * 1.5))
        url = (
            f"http://money.finance.sina.com.cn/quotes_service/api/json_v2.php"
            f"/CN_MarketData.getKLineData"
            f"?symbol={stock_code}&scale=240&datalen=1023&ma=no"
        )
        try:
            resp = requests.get(url, timeout=20, headers=_sina_headers())
            if resp.status_code in (503, 429):
                wait = 3 + random.uniform(0, 2)
                time.sleep(wait)
                break
            resp.raise_for_status()
            raw = resp.text.strip()
            if not raw or raw == "null":
                break

            rows = json.loads(raw)
            if not rows:
                break

            for row in rows:
                try:
                    d = pd.Timestamp(row["d"])
                    if d < pd.Timestamp(start_dt) or d > pd.Timestamp(end_dt):
                        continue
                    all_records.append({
                        "date":      d,
                        "open":      float(row["o"]),
                        "close":     float(row["c"]),
                        "high":      float(row["h"]),
                        "low":       float(row["l"]),
                        "volume":    float(row["v"]) * 100,
                        "amount":    float(row.get("amount", 0)),
                        "pct_change": 0.0,
                        "turnover":  0.0,
                    })
                except (KeyError, ValueError):
                    continue

            # 新浪单次返回最近 1023 条，已覆盖到请求最早日期则退出
            earliest = pd.Timestamp(rows[0]["d"])
            if earliest <= pd.Timestamp(start_dt):
                break
            cur_end = earliest - timedelta(days=1)
            _throttle(1.5)

        except Exception as e:
            logger.warning(f"[{stock_code}] 新浪拉取异常: {e}")
            break

    if not all_records:
        return pd.DataFrame()

    df = (pd.DataFrame(all_records)
            .drop_duplicates(subset=["date"])
            .sort_values("date")
            .reset_index(drop=True))
    df["pct_change"] = df["close"].pct_change().fillna(0) * 100
    return df


# ── 全A股列表（新浪接口，带本地缓存）────────────────────────────────────

def fetch_all_stock_codes(force: bool = False) -> list:
    """
    从新浪拉取全量 A 股代码列表（沪A + 深A + 科创板 + 创业板）。
    结果缓存到本地 JSON，7天内复用。

    Returns:
        list of str，格式如 ["sh600519", "sz000858", ...]
    """
    cache_file = os.path.join(DATA_CACHE_DIR, STOCK_LIST_CACHE)

    # 读取本地缓存（7天内有效）
    if not force and os.path.exists(cache_file):
        mtime = os.path.getmtime(cache_file)
        if time.time() - mtime < 7 * 86400:
            with open(cache_file, "r") as f:
                codes = json.load(f)
            if codes:
                logger.info(f"使用本地股票列表缓存，共 {len(codes)} 只")
                return codes

    # 新浪 A 股列表接口（沪A=hs_a, 深A=sz_a，含科创/创业）
    all_codes = []
    nodes = [("hs_a", "sh"), ("sz_a", "sz")]
    for node, prefix in nodes:
        page = 1
        while True:
            url = (
                f"http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php"
                f"/Market_Center.getHQNodeData"
                f"?page={page}&num=100&sort=symbol&asc=1&node={node}&_s_r_a=page"
            )
            try:
                resp = requests.get(url, timeout=15, headers=_sina_headers())
                if resp.status_code != 200 or not resp.text.strip():
                    break
                rows = json.loads(resp.text)
                if not rows:
                    break
                for r in rows:
                    sym = r.get("symbol", "")
                    if sym:
                        all_codes.append(sym)   # 新浪返回的已经是 sh600519 格式
                if len(rows) < 100:
                    break
                page += 1
                time.sleep(0.5)
            except Exception as e:
                logger.warning(f"拉取股票列表第{page}页失败: {e}")
                break

    if all_codes:
        with open(cache_file, "w") as f:
            json.dump(all_codes, f)
        logger.info(f"全A股列表已更新，共 {len(all_codes)} 只，缓存至 {cache_file}")
    else:
        # 接口失败时退回内置精选池
        logger.warning("全A股列表拉取失败，使用内置精选股票池")
        all_codes = list(STOCK_POOL)

    return all_codes


# ── 统一拉取入口（三级自动降级）─────────────────────────────────────────

def _fetch_kline(stock_code: str, start: str, end: str) -> pd.DataFrame:
    """
    拉取优先级：搜狐 → 腾讯 → 新浪
    三个源都可用，任一成功即返回。
    """
    sources = [
        ("搜狐",  _fetch_from_sohu),
        ("腾讯",  _fetch_from_tencent),
        ("新浪",  _fetch_from_sina),
    ]
    for name, fetch_fn in sources:
        try:
            df = fetch_fn(stock_code, start, end)
            if not df.empty:
                if name != "搜狐":
                    logger.info(f"[{stock_code}] 使用{name}备用源，获取 {len(df)} 条")
                return df
            logger.warning(f"[{stock_code}] {name}源无数据，尝试下一个...")
        except Exception as e:
            logger.warning(f"[{stock_code}] {name}源异常: {e}，尝试下一个...")

    logger.error(f"[{stock_code}] 三个数据源均失败")
    return pd.DataFrame()


# ── 缓存管理 ──────────────────────────────────────────────────────────

def _cache_path(stock_code: str) -> str:
    return os.path.join(DATA_CACHE_DIR, f"{stock_code}.csv")


def load_stock_data(stock_code: str, force_refresh: bool = False) -> pd.DataFrame:
    """
    加载单只股票历史数据，优先读取本地缓存，过期则增量更新。
    """
    cache_file = _cache_path(stock_code)
    today_str  = datetime.today().strftime("%Y%m%d")

    if not force_refresh and os.path.exists(cache_file):
        df = pd.read_csv(cache_file, parse_dates=["date"])
        df = df.sort_values("date").reset_index(drop=True)
        last_date  = df["date"].iloc[-1]
        days_since = (datetime.today() - last_date).days

        if days_since <= 1:
            logger.info(f"[{stock_code}] 使用缓存（最新: {last_date.date()}）")
            return df

        # 增量更新（只拉近期缺失部分）
        inc_start = (last_date + timedelta(days=1)).strftime("%Y%m%d")
        new_df = _fetch_kline(stock_code, inc_start, today_str)
        if not new_df.empty:
            df = (pd.concat([df, new_df], ignore_index=True)
                    .drop_duplicates(subset=["date"])
                    .sort_values("date")
                    .reset_index(drop=True))
            df.to_csv(cache_file, index=False)
            logger.info(f"[{stock_code}] 增量更新 {len(new_df)} 条")
        return df

    # 全量拉取
    logger.info(f"[{stock_code}] 全量拉取数据...")
    df = _fetch_kline(stock_code, START_DATE.replace("-", ""), today_str)
    if not df.empty:
        df.to_csv(cache_file, index=False)
        logger.info(f"[{stock_code}] 保存 {len(df)} 条到缓存")
    else:
        logger.warning(f"[{stock_code}] 两个数据源均失败，跳过")
    return df


def load_all_stocks(force_refresh: bool = False, quick: bool = False) -> dict:
    """
    批量加载股票数据，自动限速防止被封。

    Args:
        force_refresh: 强制重新从网络拉取
        quick: 快速模式 — 从全A股随机抽 QUICK_STOCK_COUNT 只 + 近 QUICK_HISTORY_YEARS 年
               False   — 完整模式 — 从全A股随机抽 FULL_STOCK_COUNT 只 + 完整历史
    """
    start = START_DATE.replace("-", "")

    all_codes = fetch_all_stock_codes()
    today_str = datetime.today().strftime("%Y%m%d")

    if quick:
        count = min(QUICK_STOCK_COUNT, len(all_codes))
        pool  = random.sample(all_codes, count)
        cutoff = datetime.today() - timedelta(days=QUICK_HISTORY_YEARS * 365)
        start  = cutoff.strftime("%Y%m%d")
        logger.info(f"[快速模式] 从全A股({len(all_codes)}只)随机选取 {count} 只，起始日期 {start}")
        logger.info(f"[快速模式] 股票列表: {pool}")
    else:
        count = min(FULL_STOCK_COUNT, len(all_codes))
        pool  = random.sample(all_codes, count)
        logger.info(f"[完整模式] 从全A股({len(all_codes)}只)随机选取 {count} 只，起始日期 {start}")
        logger.info(f"[完整模式] 股票列表: {pool}")

    result = {}
    total  = len(pool)
    for idx, code in enumerate(pool, 1):
        name = get_stock_name(code)
        print(f"  [{idx:>3}/{total}] {code}  {name:<8} 拉取中...", flush=True)
        logger.info(f"正在加载 ({idx}/{total}): {code} {name}")
        try:
            if quick:
                df = _fetch_kline(code, start, today_str)
            else:
                df = load_stock_data(code, force_refresh=force_refresh)

            if not df.empty:
                result[code] = df
                print(f"  [{idx:>3}/{total}] {code}  {name:<8} ✓ {len(df)} 条")
                logger.info(f"[{code}] {name} ✓ {len(df)} 条")
            else:
                print(f"  [{idx:>3}/{total}] {code}  {name:<8} ✗ 无数据")
                logger.warning(f"[{code}] {name} ✗ 三个源均无数据")
        except Exception as e:
            logger.error(f"[{code}] 加载异常: {e}")

        # 动态限速：每5只股票后稍长休息，避免持续高频请求
        if idx % 5 == 0:
            wait = random.uniform(3, 5)
            logger.info(f"已完成 {idx}/{total}，休息 {wait:.1f}s 避免限流...")
            time.sleep(wait)
        else:
            _throttle(1.2)   # 正常间隔 0.6~1.8s

    logger.info(f"数据加载完成：{len(result)}/{total} 只成功")
    return result


def get_stock_name(stock_code: str) -> str:
    """通过新浪行情获取股票名称（字段[0]）。"""
    try:
        url  = f"https://hq.sinajs.cn/list={stock_code}"
        resp = requests.get(url, timeout=10, headers=_SINA_HEADERS)
        text = resp.content.decode("gb18030", errors="replace")
        m    = re.search(r'"([^"]*)"', text)
        if m:
            fields = m.group(1).split(",")
            if fields and fields[0].strip():
                return fields[0].strip()
    except Exception:
        pass
    return stock_code
