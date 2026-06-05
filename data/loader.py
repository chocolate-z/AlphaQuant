# 数据加载：搜狐前复权日K线（主）+ 腾讯前复权K线（备）+ 本地CSV缓存

import os
import re
import json
import time
import random
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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

    # 搜狐已降级为备用源（腾讯为主），故快速失败：2 次、短等待，避免单只股票
    # 卡在 503 退避上拖慢整体（旧实现 4 次退避最坏要等约 45 秒）。
    for attempt in range(2):
        try:
            resp = requests.get(url, timeout=15, headers=_sohu_headers(), allow_redirects=True)
            if resp.status_code in (503, 429):
                wait = 1.5 * (attempt + 1) + random.uniform(0, 1)
                logger.warning(f"[{stock_code}] 搜狐 {resp.status_code} 限流，{wait:.1f}s 后重试（第{attempt+1}/2次，备用源）")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            break
        except requests.exceptions.RequestException as e:
            if attempt == 1:
                logger.warning(f"[{stock_code}] 搜狐连接失败: {e}")
                return pd.DataFrame()
            time.sleep(1.5)
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

# 腾讯 fqkline 单次 maxBars 上限实测 ≈ 800（>800 会被悄悄截断到 640，>2000 直接 param error）。
# 故按 end 向前翻页拼接，覆盖完整历史。这是本项目最稳定的日线源。
_TENCENT_MAXBARS = 800


def _fetch_tencent_page(stock_code: str, start_fmt: str, end_fmt: str) -> list:
    """
    取腾讯 fqkline 的一页（最多 _TENCENT_MAXBARS 个交易日，结束于 end_fmt）。
    返回原始行列表 [[日期,开,收,高,低,量], ...]；失败/无数据返回 []。
    关键：param error 时 data 是空 list 而非 dict，必须容错（旧实现就栽在这）。
    """
    url = (
        f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
        f"?param={stock_code},day,{start_fmt},{end_fmt},{_TENCENT_MAXBARS},qfq"
    )
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=20, headers=_tencent_headers())
            if resp.status_code in (503, 429):
                time.sleep(2 * (2 ** attempt) + random.uniform(0, 1))
                continue
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") != 0:
                return []
            payload = data.get("data")
            if not isinstance(payload, dict):   # param error → data 为 [] ，容错
                return []
            stock_data = payload.get(stock_code) or {}
            if not isinstance(stock_data, dict):
                return []
            return stock_data.get("qfqday") or stock_data.get("day") or []
        except requests.exceptions.RequestException as e:
            if attempt == 2:
                logger.warning(f"[{stock_code}] 腾讯连接失败: {e}")
                return []
            time.sleep(2 ** attempt)
        except Exception as e:
            logger.warning(f"[{stock_code}] 腾讯解析失败: {e}")
            return []
    return []


def _fetch_from_tencent(stock_code: str, start: str, end: str) -> pd.DataFrame:
    """
    腾讯前复权日K线（**主源**，境外可达、稳定）。
    接口: web.ifzq.gtimg.cn/appstock/app/fqkline/get（按 end 向前翻页取全历史）
    字段: [日期, 开盘, 收盘, 最高, 最低, 成交量(手)]
    注：腾讯不含成交额和换手率，填 0（特征层的换手率列在缓存历史里由搜狐提供）。
    """
    start_dt = datetime.strptime(start, "%Y%m%d")
    start_fmt = start_dt.strftime("%Y-%m-%d")
    cur_end   = datetime.strptime(end, "%Y%m%d")

    merged: dict = {}     # 日期字符串 → 原始行，天然去重（翻页有 1 天重叠）
    for _page in range(12):   # 800×12≈9600 天，远超 A 股最长历史
        end_fmt = cur_end.strftime("%Y-%m-%d")
        rows = _fetch_tencent_page(stock_code, start_fmt, end_fmt)
        if not rows:
            break
        for row in rows:
            merged[row[0]] = row
        earliest = pd.Timestamp(rows[0][0])
        if earliest <= pd.Timestamp(start_dt):
            break
        nxt = earliest - pd.Timedelta(days=1)
        if nxt >= cur_end:        # 没有向前推进，防死循环
            break
        cur_end = nxt.to_pydatetime()
        _throttle(0.5)

    if not merged:
        return pd.DataFrame()

    end_ts = pd.Timestamp(datetime.strptime(end, "%Y%m%d"))
    start_ts = pd.Timestamp(start_dt)
    records = []
    for row in merged.values():
        try:
            d = pd.Timestamp(row[0])
            if d < start_ts or d > end_ts:
                continue
            records.append({
                "date":   d,
                "open":   float(row[1]),
                "close":  float(row[2]),
                "high":   float(row[3]),
                "low":    float(row[4]),
                "volume": float(row[5]) * 100,   # 手 → 股
                "amount": 0.0,
                "pct_change": 0.0,               # 下方按收盘价推算
                "turnover":  0.0,
            })
        except (ValueError, IndexError):
            continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    df["pct_change"] = df["close"].pct_change().fillna(0) * 100
    return df


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
    拉取优先级（按板块自适应，任一源成功即返回）：
      · 沪深/科创/创业（sh/sz）：腾讯 → 搜狐 → 新浪（腾讯 fqkline 最稳、不限流、可取全历史）
      · 北交所（bj）：搜狐 → 腾讯 → 新浪（腾讯 fqkline 不覆盖北交所历史，只返回 1 条）
    """
    if stock_code.startswith("bj"):
        sources = [("搜狐", _fetch_from_sohu), ("腾讯", _fetch_from_tencent), ("新浪", _fetch_from_sina)]
    else:
        sources = [("腾讯", _fetch_from_tencent), ("搜狐", _fetch_from_sohu), ("新浪", _fetch_from_sina)]
    primary = sources[0][0]

    for name, fetch_fn in sources:
        try:
            df = fetch_fn(stock_code, start, end)
            if not df.empty:
                if name != primary:
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


def load_cached_stocks(start: str = None, end: str = None,
                       min_rows: int = 60, limit: int = None,
                       boards: tuple = ("sh", "sz", "bj")) -> dict:
    """
    直接读取本地缓存目录下已下载的个股 CSV —— 完全离线，不联网。

    适合本机已积累大量缓存的场景：数据更多、速度快、不会触发搜狐 503 限流，
    且结果可复现。会自动跳过非个股缓存文件（market_features.csv 等）。

    Args:
        start/end: 可选日期区间 YYYYMMDD，提供时按区间过滤
        min_rows:  少于该行数的股票跳过（数据太少没法构特征）
        limit:     最多加载多少只（None=全部）。超过 limit 时**均匀抽样**而非只取前 N，
                   覆盖面更广、更有代表性
        boards:    保留哪些板块前缀，默认沪深+北交所。训练正式模型建议只用 ("sh","sz")：
                   北交所(bj)流动性差、走势特殊，混进来会拖累模型质量

    Returns:
        {code: DataFrame}
    """
    import glob
    files = sorted(glob.glob(os.path.join(DATA_CACHE_DIR, "*.csv")))
    prefixes = tuple(boards)

    codes = []
    for fp in files:
        name = os.path.splitext(os.path.basename(fp))[0]
        # 仅保留形如 sh600519 / sz000858 / bj920000，且属于指定板块的个股缓存
        if re.match(r"^(sh|sz|bj)\d{6}$", name) and name.startswith(prefixes):
            codes.append((name, fp))

    if limit is not None and len(codes) > limit:
        # 均匀抽样：跨整个代码区间取 limit 只，避免只取前 N 段（更有代表性）
        stepf = len(codes) / float(limit)
        codes = [codes[int(i * stepf)] for i in range(limit)]

    start_dt = pd.Timestamp(datetime.strptime(start, "%Y%m%d")) if start else None
    end_dt   = pd.Timestamp(datetime.strptime(end,   "%Y%m%d")) if end else None

    result: dict = {}
    for code, fp in codes:
        try:
            df = pd.read_csv(fp, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
            if start_dt is not None:
                df = df[df["date"] >= start_dt]
            if end_dt is not None:
                df = df[df["date"] <= end_dt]
            df = df.reset_index(drop=True)
            if len(df) >= min_rows:
                result[code] = df
        except Exception as e:
            logger.warning(f"[{code}] 读取缓存失败: {e}")

    logger.info(f"[缓存池] 从本地缓存加载 {len(result)} 只股票"
                f"（扫描到 {len(codes)} 个个股缓存文件）")
    return result


def get_tradeable_pool(limit: int = 100, recent_days: int = 250,
                       min_amount_yi: float = 2.0, boards: tuple = ("sh", "sz")) -> list:
    """
    构建「可交易股票池」：从本地缓存里按近 recent_days 日成交额中位数排序，
    取流动性最好的 limit 只（且中位成交额 ≥ min_amount_yi 亿元）。

    为什么需要它：**回测和实盘必须用同一批、且流动性好的股票**，否则回测里能成交、
    实盘却买不进卖不出，收益不可比。蓝筹流动性好、滑点小、也极少退市（顺带缓解幸存者偏差）。

    Returns:
        list[str]：代码，按流动性从高到低排序
    """
    import glob
    files = sorted(glob.glob(os.path.join(DATA_CACHE_DIR, "*.csv")))
    prefixes = tuple(boards)
    scored = []
    for fp in files:
        name = os.path.splitext(os.path.basename(fp))[0]
        if not (re.match(r"^(sh|sz|bj)\d{6}$", name) and name.startswith(prefixes)):
            continue
        try:
            df = pd.read_csv(fp, usecols=["amount"]).tail(recent_days)
            if len(df) < 60:
                continue
            amt = pd.to_numeric(df["amount"], errors="coerce").median()
            if amt and amt > 0:
                scored.append((name, float(amt)))
        except Exception:
            continue
    scored.sort(key=lambda x: -x[1])
    thr = min_amount_yi * 1e8
    pool = [c for c, a in scored if a >= thr][:limit]
    if len(pool) < 20:                       # 阈值太严时兜底：直接取流动性最高的 limit 只
        pool = [c for c, _ in scored][:limit]
    logger.info(f"[可交易池] 选出 {len(pool)} 只流动性最好的股票"
                f"（近 {recent_days} 日成交额中位数 ≥ {min_amount_yi} 亿）")
    return pool


def load_all_stocks(force_refresh: bool = False, quick: bool = False,
                    start: str = None, end: str = None) -> dict:
    """
    批量加载股票数据，并发4线程 + 信号量限速防止被封。

    Args:
        force_refresh: 强制重新从网络拉取
        quick: 快速模式 — 从全A股随机抽 QUICK_STOCK_COUNT 只 + 近 QUICK_HISTORY_YEARS 年
               False   — 完整模式 — 从全A股随机抽 FULL_STOCK_COUNT 只 + 完整历史
        start: 可选起始日期 YYYYMMDD，提供时对所有股票使用 _fetch_kline
        end:   可选截止日期 YYYYMMDD
    """
    default_start = START_DATE.replace("-", "")

    all_codes = fetch_all_stock_codes()
    today_str = datetime.today().strftime("%Y%m%d")

    if quick:
        count = min(QUICK_STOCK_COUNT, len(all_codes))
        pool  = random.sample(all_codes, count)
        cutoff = datetime.today() - timedelta(days=QUICK_HISTORY_YEARS * 365)
        fetch_start = start or cutoff.strftime("%Y%m%d")
        logger.info(f"[快速模式] 从全A股({len(all_codes)}只)随机选取 {count} 只，起始日期 {fetch_start}")
        logger.info(f"[快速模式] 股票列表: {pool}")
    else:
        count = min(FULL_STOCK_COUNT, len(all_codes))
        pool  = random.sample(all_codes, count)
        fetch_start = start or default_start
        logger.info(f"[完整模式] 从全A股({len(all_codes)}只)随机选取 {count} 只，起始日期 {fetch_start}")
        logger.info(f"[完整模式] 股票列表: {pool}")

    fetch_end = end or today_str
    use_kline = quick or (start is not None)  # custom date range forces kline path

    result: dict = {}
    total = len(pool)
    _print_lock = threading.Lock()
    _count_lock = threading.Lock()
    _semaphore = threading.Semaphore(4)
    _completed = [0]  # mutable counter

    def _fetch_one(args):
        idx, code = args
        name = get_stock_name(code)
        with _print_lock:
            print(f"  [{idx:>3}/{total}] {code}  {name:<8} 拉取中...", flush=True)
        logger.info(f"正在加载 ({idx}/{total}): {code} {name}")

        _semaphore.acquire()
        try:
            time.sleep(random.uniform(0.3, 1.0))  # per-thread random delay
            if use_kline:
                df = _fetch_kline(code, fetch_start, fetch_end)
            else:
                df = load_stock_data(code, force_refresh=force_refresh)
        finally:
            _semaphore.release()

        if df is not None and not df.empty:
            with _print_lock:
                print(f"  [{idx:>3}/{total}] {code}  {name:<8} ✓ {len(df)} 条")
            logger.info(f"[{code}] {name} ✓ {len(df)} 条")
            return code, df
        else:
            with _print_lock:
                print(f"  [{idx:>3}/{total}] {code}  {name:<8} ✗ 无数据")
            logger.warning(f"[{code}] {name} ✗ 三个源均无数据")
            return code, None

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_fetch_one, (idx, code)): code
                   for idx, code in enumerate(pool, 1)}
        for future in as_completed(futures):
            try:
                code, df = future.result()
                if df is not None:
                    result[code] = df
            except Exception as e:
                logger.error(f"加载异常: {e}")

            with _count_lock:
                _completed[0] += 1
                done = _completed[0]

            # 每完成5只，稍作休息
            if done % 5 == 0:
                wait = random.uniform(2, 3)
                logger.info(f"已完成 {done}/{total}，休息 {wait:.1f}s 避免限流...")
                time.sleep(wait)

    logger.info(f"数据加载完成：{len(result)}/{total} 只成功")
    return result


_name_cache: dict = {}

def get_stock_name(stock_code: str) -> str:
    """通过新浪行情获取股票名称（字段[0]）。"""
    if stock_code in _name_cache:
        return _name_cache[stock_code]
    try:
        url  = f"https://hq.sinajs.cn/list={stock_code}"
        resp = requests.get(url, timeout=10, headers=_sina_headers())
        text = resp.content.decode("gb18030", errors="replace")
        m    = re.search(r'"([^"]*)"', text)
        if m:
            fields = m.group(1).split(",")
            if fields and fields[0].strip():
                name = fields[0].strip()
                _name_cache[stock_code] = name
                return name
    except Exception:
        pass
    _name_cache[stock_code] = stock_code
    return stock_code
