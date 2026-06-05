# 特征构建：16个技术特征 + 逐窗口Z-Score归一化，解决跨股票尺度问题

import os
import time
import logging
import numpy as np
import pandas as pd
import joblib

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (WINDOW_SIZE, LABEL_HORIZON, LABEL_THRESHOLD, MODEL_SAVE_DIR,
                    TRAIN_RATIO, DATA_CACHE_DIR, START_DATE,
                    USE_EXCESS_LABEL, EXCESS_THRESHOLD, USE_CROSS_SECTIONAL)

logger = logging.getLogger(__name__)

# 16个特征：收益动量、量能、技术形态、均线偏离、震荡指标
FEATURE_NAMES = [
    # ── 价格动量 ──
    "ret_1d",       # 当日收益率（日涨跌幅）
    "ret_5d",       # 5日累积收益率
    "ret_20d",      # 20日累积收益率
    # ── 均线偏离 ──
    "ma5_dev",      # 收盘价偏离5日均线的百分比
    "ma10_dev",     # 收盘价偏离10日均线的百分比
    "ma20_dev",     # 收盘价偏离20日均线的百分比
    "ma60_dev",     # 收盘价偏离60日均线的百分比
    # ── 量能 ──
    "vol_ratio",    # 当日成交量 / 20日均量（量比）
    "vol_trend",    # 5日均量 / 20日均量（量能趋势）
    "turnover",     # 换手率
    # ── 价格形态 ──
    "amplitude",    # 振幅：(最高-最低)/昨收
    "close_strength", # 收盘强度：(收-低)/(高-低)，1=收于最高
    "open_gap",     # 跳空幅度：开盘/昨收-1
    # ── 震荡指标 ──
    "rsi14",        # RSI(14)，归一化到[0,1]
    "macd_hist",    # MACD柱状图（归一化）
    "bb_pos",       # 布林带位置：(收-下轨)/(上轨-下轨)，越高越强
]

# ── 市场环境特征（沪深300，全市场同日相同，编码大盘状态）──
# 让模型知道"当下大盘环境"，区分个股涨是普涨还是逆势走强
MARKET_FEATURES = [
    "mkt_ret_1d",    # 大盘当日涨跌幅
    "mkt_ret_5d",    # 大盘5日累计涨跌幅
    "mkt_ma20_dev",  # 大盘偏离20日均线（牛熊环境）
]

# ── 横截面特征（机构级 alpha 技术）──
# 同一天里，这只票的动量/量能/RSI 在**全市场**排第几名（百分位 0~1）。
# 这是"相对强弱"——新增信息，比再堆价量指标更能提升选股 alpha。
# 训练/回测/实盘都在各自的股票池内计算排名（口径一致）。
CROSS_FEATURES = [
    "cs_ret5d",     # 5日收益的全市场百分位排名
    "cs_ret20d",    # 20日收益的全市场百分位排名
    "cs_volratio",  # 量比的全市场百分位排名
    "cs_rsi",       # RSI 的全市场百分位排名
] if USE_CROSS_SECTIONAL else []     # 默认关闭（实测净减分）

FEATURE_NAMES = FEATURE_NAMES + MARKET_FEATURES + CROSS_FEATURES

FEATURE_DIM = len(FEATURE_NAMES)  # 23

# 沪深300指数缓存（惰性加载，所有路径共享）
_MARKET_CACHE = None


def _load_market_df() -> pd.DataFrame:
    """
    惰性加载沪深300指数并计算市场环境特征，结果缓存到内存 + 磁盘（3天有效）。
    所有调用 compute_raw_features 的路径（训练/推理/回测/诊断）自动共享。
    返回 DataFrame[date, mkt_ret_1d, mkt_ret_5d, mkt_ma20_dev]；失败返回空表。
    """
    global _MARKET_CACHE
    if _MARKET_CACHE is not None:
        return _MARKET_CACHE

    cache_file = os.path.join(DATA_CACHE_DIR, "market_features.csv")
    # 7天内的缓存直接使用
    if os.path.exists(cache_file) and time.time() - os.path.getmtime(cache_file) < 7 * 86400:
        try:
            _MARKET_CACHE = pd.read_csv(cache_file, parse_dates=["date"])
            logger.info(f"市场特征使用缓存（{len(_MARKET_CACHE)} 条交易日）")
            return _MARKET_CACHE
        except Exception:
            pass

    try:
        from data.index_fetcher import fetch_index_history
        from datetime import datetime
        idx = fetch_index_history("cn_s_sh000300",
                                  START_DATE.replace("-", ""),
                                  datetime.today().strftime("%Y%m%d"))
        if idx is None or idx.empty:
            # 降级：尝试使用过期的旧缓存
            if os.path.exists(cache_file):
                try:
                    stale = pd.read_csv(cache_file, parse_dates=["date"])
                    if not stale.empty:
                        logger.warning("沪深300在线拉取失败，使用过期缓存（市场特征可能略旧）")
                        _MARKET_CACHE = stale
                        return _MARKET_CACHE
                except Exception:
                    pass
            logger.warning("沪深300指数拉取失败且无缓存，市场特征将填 0")
            _MARKET_CACHE = pd.DataFrame(columns=["date"] + MARKET_FEATURES)
            return _MARKET_CACHE

        idx = idx.sort_values("date").reset_index(drop=True)
        c = idx["close"]
        out = pd.DataFrame({"date": idx["date"]})
        out["mkt_ret_1d"]   = c.pct_change().fillna(0).clip(-0.1, 0.1)
        out["mkt_ret_5d"]   = c.pct_change(5).fillna(0).clip(-0.2, 0.2)
        ma20 = c.rolling(20, min_periods=1).mean()
        out["mkt_ma20_dev"] = (c / ma20 - 1).fillna(0).clip(-0.2, 0.2)
        out["mkt_close"]    = c.values   # 原始收盘价，供超额收益标签使用（不进 FEATURE_NAMES）

        out.to_csv(cache_file, index=False)
        logger.info(f"市场特征已加载（沪深300，{len(out)} 个交易日）")
        _MARKET_CACHE = out
        return out
    except Exception as e:
        logger.warning(f"市场特征加载异常: {e}，将填 0")
        _MARKET_CACHE = pd.DataFrame(columns=["date"] + MARKET_FEATURES)
        return _MARKET_CACHE


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(period, min_periods=1).mean()
    rs = gain / (loss + 1e-9)
    return rs / (1 + rs)   # 归一化到 [0, 1]


def _macd_hist(series: pd.Series) -> pd.Series:
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd  = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    # 用20日滚动std归一化，保持量纲一致
    std = hist.rolling(20, min_periods=1).std().replace(0, 1)
    return hist / std


def compute_raw_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算16个技术特征。所有特征均为相对量（百分比/比率），
    不含绝对价格，天然支持跨股票通用。
    """
    df = df.copy().reset_index(drop=True)

    for col in ["open", "close", "high", "low", "volume", "pct_change", "turnover"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    close = df["close"]
    prev  = close.shift(1).fillna(close)

    # ── 收益动量 ──────────────────────────────────────
    df["ret_1d"]  = (close / prev - 1).fillna(0)
    df["ret_5d"]  = close.pct_change(5).fillna(0)
    df["ret_20d"] = close.pct_change(20).fillna(0)

    # ── 均线偏离 ──────────────────────────────────────
    for n, col in [(5, "ma5_dev"), (10, "ma10_dev"), (20, "ma20_dev"), (60, "ma60_dev")]:
        ma = close.rolling(n, min_periods=1).mean()
        df[col] = (close / ma - 1).fillna(0)

    # ── 量能 ──────────────────────────────────────────
    vol     = df["volume"].replace(0, np.nan).ffill().fillna(1)
    vol_ma5  = vol.rolling(5,  min_periods=1).mean()
    vol_ma20 = vol.rolling(20, min_periods=1).mean()
    df["vol_ratio"]  = (vol / (vol_ma20 + 1e-9)).clip(0, 10)
    df["vol_trend"]  = (vol_ma5 / (vol_ma20 + 1e-9)).clip(0, 5)
    df["turnover"]   = df["turnover"].clip(0, 30)

    # ── 价格形态 ──────────────────────────────────────
    df["amplitude"]     = ((df["high"] - df["low"]) / (prev + 1e-9)).clip(0, 0.2)
    hl = (df["high"] - df["low"]).replace(0, np.nan)
    df["close_strength"] = ((close - df["low"]) / hl).fillna(0.5).clip(0, 1)
    df["open_gap"]      = (df["open"] / (prev + 1e-9) - 1).fillna(0).clip(-0.1, 0.1)

    # ── 震荡指标 ──────────────────────────────────────
    df["rsi14"]     = _rsi(close, 14)
    df["macd_hist"] = _macd_hist(close)

    # 布林带位置（20日，2倍标准差）
    ma20 = close.rolling(20, min_periods=1).mean()
    std20 = close.rolling(20, min_periods=1).std().fillna(1)
    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20
    band  = (upper - lower).replace(0, np.nan)
    df["bb_pos"] = ((close - lower) / band).fillna(0.5).clip(0, 1)

    # ── 市场环境特征（按日期 merge 沪深300，缺失填 0）──────
    mkt = _load_market_df()
    if mkt is not None and not mkt.empty and "date" in df.columns:
        df = df.merge(mkt, on="date", how="left")
    for col in MARKET_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df


def compute_cross_sectional(stock_data_dict: dict) -> dict:
    """
    计算「横截面排名」特征：对每个交易日，把池内所有股票的某指标排序，取百分位（0~1）。
    例如某天某票 cs_ret5d=0.95，表示它的 5 日动量强过全池 95% 的票。

    口径一致性：训练/回测/实盘各自在自己的股票池内计算，保证排名含义一致。

    Args:
        stock_data_dict: {code: 原始日线 DataFrame}
    Returns:
        {code: DataFrame[date, cs_ret5d, cs_ret20d, cs_volratio, cs_rsi]}
    """
    src = [("ret_5d", "cs_ret5d"), ("ret_20d", "cs_ret20d"),
           ("vol_ratio", "cs_volratio"), ("rsi14", "cs_rsi")]
    frames = []
    for code, df in stock_data_dict.items():
        if df is None or "date" not in df.columns or len(df) < WINDOW_SIZE:
            continue
        try:
            r = compute_raw_features(df)
            sub = r[["date"] + [s for s, _ in src]].copy()
            sub["code"] = code
            frames.append(sub)
        except Exception:
            continue
    if not frames:
        return {}
    big = pd.concat(frames, ignore_index=True)
    for raw, cs in src:
        # 每个交易日内做百分位排名（同名次取均值），缺失记 0.5（中性）
        big[cs] = big.groupby("date")[raw].rank(pct=True, method="average")
    cols = ["date"] + [c for _, c in src]
    return {code: g[cols].reset_index(drop=True) for code, g in big.groupby("code")}


def _window_zscore(window: np.ndarray) -> np.ndarray:
    """
    逐窗口 Z-Score 归一化：每个特征在该窗口内减去均值除以标准差。
    解决不同股票价格尺度差异，让模型学习相对变化规律。
    """
    mean = window.mean(axis=0, keepdims=True)
    std  = window.std(axis=0, keepdims=True)
    std  = np.where(std < 1e-8, 1.0, std)
    return (window - mean) / std


def build_sequences(df: pd.DataFrame, scaler=None, fit_scaler: bool = False, cs_df=None):  # scaler/fit_scaler unused, kept for API compat
    """
    将特征 DataFrame 转换为 LSTM 输入序列和标签。
    全向量化实现（无 Python 循环），比逐窗口循环快 20~50x。

    cs_df: 该股票的横截面排名特征（来自 compute_cross_sectional），按 date 合并；
           不提供时横截面特征填 0.5（中性）——单股推理无全市场口径时如此降级。
    """
    df = compute_raw_features(df)
    # 合并横截面排名特征（缺失填 0.5 中性，保证特征维度恒为 FEATURE_DIM）
    if cs_df is not None and "date" in df.columns:
        df = df.merge(cs_df, on="date", how="left")
    for col in CROSS_FEATURES:
        if col not in df.columns:
            df[col] = 0.5
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.5)

    available = [c for c in FEATURE_NAMES if c in df.columns]
    feat = df[available].values.astype(np.float32)
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

    closes = df["close"].values
    highs  = df["high"].values
    n = len(df)

    # ── 标签向量化 ────────────────────────────────────────────────────
    from numpy.lib.stride_tricks import sliding_window_view
    labels      = np.zeros(n, dtype=np.float32)
    cur_close   = closes[:n - LABEL_HORIZON]          # closes[0..n-H-1]
    fwd_close   = closes[LABEL_HORIZON:]              # closes[H..n-1]

    if USE_EXCESS_LABEL and "mkt_close" in df.columns:
        # 超额收益标签：个股5日涨幅 − 大盘5日涨幅 > EXCESS_THRESHOLD
        # 剥离大盘 beta，让模型专注个股相对强弱（alpha）
        mkt_c     = df["mkt_close"].values.astype(float)
        mkt_cur   = mkt_c[:n - LABEL_HORIZON]
        mkt_fwd   = mkt_c[LABEL_HORIZON:]
        stock_ret = np.where(cur_close  > 0, fwd_close / cur_close  - 1, 0.0)
        mkt_ret   = np.where(mkt_cur    > 0, mkt_fwd   / mkt_cur    - 1, 0.0)
        excess    = stock_ret - mkt_ret
        labels[:n - LABEL_HORIZON] = (excess > EXCESS_THRESHOLD).astype(np.float32)
    else:
        # 原始标签：未来 H 日内最高价涨幅 > LABEL_THRESHOLD（绝对涨幅）
        future     = sliding_window_view(highs, LABEL_HORIZON)[1:]  # (n-H, H)
        future_max = future.max(axis=1)
        mask       = cur_close > 0
        labels[:n - LABEL_HORIZON][mask] = (
            (future_max[mask] - cur_close[mask]) / cur_close[mask] > LABEL_THRESHOLD
        ).astype(np.float32)

    # ── 序列向量化 ───────────────────────────────────────────────────
    # sliding_window_view: (N, WINDOW_SIZE, FEATURE_DIM)，零拷贝
    windows = sliding_window_view(feat, (WINDOW_SIZE, feat.shape[1])
                                  ).reshape(-1, WINDOW_SIZE, feat.shape[1])
    # 有效索引：[WINDOW_SIZE, n-LABEL_HORIZON)
    start_idx = WINDOW_SIZE
    end_idx   = n - LABEL_HORIZON
    X = windows[start_idx - WINDOW_SIZE: end_idx - WINDOW_SIZE].copy()  # (N, W, F)
    y = labels[start_idx: end_idx]

    # ── 逐窗口 Z-Score（向量化）──────────────────────────────────────
    mean = X.mean(axis=1, keepdims=True)          # (N, 1, F)
    std  = X.std(axis=1, keepdims=True)           # (N, 1, F)
    std  = np.where(std < 1e-8, 1.0, std)
    X    = (X - mean) / std

    dates = (df["date"].iloc[start_idx:end_idx].tolist()
             if "date" in df.columns else list(range(start_idx, end_idx)))

    return X, y, None, dates


def build_all_stocks(stock_data_dict: dict, fit_scaler: bool = True):  # fit_scaler unused, kept for API compat
    """
    对所有股票构建特征序列并合并。
    逐窗口归一化无需全局 scaler，fit_scaler 参数保留用于接口兼容。

    同时返回每条序列对应的「日期」数组，供训练器做**真正的时序切分**
    （用较早的数据训练、用较晚的数据验证，真实检验"用历史预测未来"的能力）。
    注意：合并后的数组是「按股票」拼接的，本身并非按时间排序，
    所以必须依赖 dates 数组才能正确切分，不能简单按位置切。

    Returns:
        X:      (N, WINDOW_SIZE, FEATURE_DIM)
        y:      (N,)
        scaler: None
        dates:  (N,) datetime64[ns]，与 X/y 行对齐的样本日期
    """
    all_X, all_y, all_dates = [], [], []

    # 先在整个股票池上算横截面排名（开启时），再逐股构建序列
    if CROSS_FEATURES:
        logger.info("计算横截面排名特征（全市场相对强弱）...")
        cs_map = compute_cross_sectional(stock_data_dict)
    else:
        cs_map = {}

    for code, df in stock_data_dict.items():
        if len(df) < WINDOW_SIZE + LABEL_HORIZON + 10:
            logger.warning(f"[{code}] 数据太少（{len(df)}行），跳过")
            continue
        try:
            X, y, _, dts = build_sequences(df, cs_df=cs_map.get(code))
            if len(X) > 0:
                all_X.append(X)
                all_y.append(y)
                all_dates.append(pd.to_datetime(dts).values.astype("datetime64[ns]"))
                logger.debug(f"[{code}] 构建 {len(X)} 条序列，正样本 {y.mean():.2%}")
        except Exception as e:
            logger.error(f"[{code}] 特征构建失败: {e}")

    if not all_X:
        return np.array([]), np.array([]), None, np.array([], dtype="datetime64[ns]")

    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    dates_all = np.concatenate(all_dates)

    # 保存一个占位 scaler 文件，保持接口兼容
    scaler_path = os.path.join(MODEL_SAVE_DIR, "scaler.joblib")
    joblib.dump({"type": "window_zscore", "feature_dim": FEATURE_DIM}, scaler_path)

    logger.info(f"共构建 {len(X_all)} 条序列，正样本比例 {y_all.mean():.3f}")
    return X_all, y_all, None, dates_all


def load_scaler():
    """兼容接口，逐窗口归一化不需要全局 scaler。"""
    return None


def build_inference_sequence(df: pd.DataFrame, scaler=None, cs_df=None) -> np.ndarray:
    """
    为推理构建最新一个序列（逐窗口归一化）。
    cs_df：该股票横截面排名特征；不提供则填 0.5（单股推理无全市场口径时降级）。

    Returns:
        X: np.ndarray (1, WINDOW_SIZE, FEATURE_DIM)
    """
    df = compute_raw_features(df)
    if cs_df is not None and "date" in df.columns:
        df = df.merge(cs_df, on="date", how="left")
    for col in CROSS_FEATURES:
        if col not in df.columns:
            df[col] = 0.5
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.5)
    available = [c for c in FEATURE_NAMES if c in df.columns]
    feat = df[available].values.astype(np.float32)
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

    if len(feat) < WINDOW_SIZE:
        raise ValueError(f"数据行数不足 {WINDOW_SIZE}，无法构建推理序列")

    window = feat[-WINDOW_SIZE:].copy()
    window = _window_zscore(window)
    return window[np.newaxis, :, :]   # (1, WINDOW_SIZE, FEATURE_DIM)
