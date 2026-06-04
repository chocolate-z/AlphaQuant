# 特征构建：16个技术特征 + 逐窗口Z-Score归一化，解决跨股票尺度问题

import os
import logging
import numpy as np
import pandas as pd
import joblib

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import WINDOW_SIZE, LABEL_HORIZON, LABEL_THRESHOLD, MODEL_SAVE_DIR, TRAIN_RATIO

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

FEATURE_DIM = len(FEATURE_NAMES)  # 16


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

    return df


def _window_zscore(window: np.ndarray) -> np.ndarray:
    """
    逐窗口 Z-Score 归一化：每个特征在该窗口内减去均值除以标准差。
    解决不同股票价格尺度差异，让模型学习相对变化规律。
    """
    mean = window.mean(axis=0, keepdims=True)
    std  = window.std(axis=0, keepdims=True)
    std  = np.where(std < 1e-8, 1.0, std)
    return (window - mean) / std


def build_sequences(df: pd.DataFrame, scaler=None, fit_scaler: bool = False):
    """
    将特征 DataFrame 转换为 LSTM 输入序列和标签。
    使用逐窗口 Z-Score 归一化（scaler参数保留用于接口兼容）。

    Returns:
        X: np.ndarray (N, WINDOW_SIZE, FEATURE_DIM)
        y: np.ndarray (N,)
        scaler: None（逐窗口归一化不需要全局scaler）
        dates: list
    """
    df = compute_raw_features(df)
    available = [c for c in FEATURE_NAMES if c in df.columns]
    feat = df[available].values.astype(np.float32)
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

    closes = df["close"].values
    highs  = df["high"].values
    n = len(df)

    labels = np.zeros(n, dtype=np.float32)
    for i in range(n - LABEL_HORIZON):
        future_high = highs[i + 1: i + 1 + LABEL_HORIZON].max()
        if closes[i] > 0:
            labels[i] = 1.0 if (future_high - closes[i]) / closes[i] > LABEL_THRESHOLD else 0.0

    X, y, dates = [], [], []
    for i in range(WINDOW_SIZE, n - LABEL_HORIZON):
        window = feat[i - WINDOW_SIZE: i].copy()
        window = _window_zscore(window)
        X.append(window)
        y.append(labels[i])
        dates.append(df["date"].iloc[i] if "date" in df.columns else i)

    return np.array(X), np.array(y), None, dates


def build_all_stocks(stock_data_dict: dict, fit_scaler: bool = True):
    """
    对所有股票构建特征序列并合并。
    逐窗口归一化无需全局 scaler，fit_scaler 参数保留用于接口兼容。

    Returns:
        X: (N, WINDOW_SIZE, FEATURE_DIM)
        y: (N,)
        scaler: None
    """
    all_X, all_y = [], []

    for code, df in stock_data_dict.items():
        if len(df) < WINDOW_SIZE + LABEL_HORIZON + 10:
            logger.warning(f"[{code}] 数据太少（{len(df)}行），跳过")
            continue
        try:
            X, y, _, _ = build_sequences(df)
            if len(X) > 0:
                all_X.append(X)
                all_y.append(y)
                logger.debug(f"[{code}] 构建 {len(X)} 条序列，正样本 {y.mean():.2%}")
        except Exception as e:
            logger.error(f"[{code}] 特征构建失败: {e}")

    if not all_X:
        return np.array([]), np.array([]), None

    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)

    # 保存一个占位 scaler 文件，保持接口兼容
    scaler_path = os.path.join(MODEL_SAVE_DIR, "scaler.joblib")
    joblib.dump({"type": "window_zscore", "feature_dim": FEATURE_DIM}, scaler_path)

    logger.info(f"共构建 {len(X_all)} 条序列，正样本比例 {y_all.mean():.3f}")
    return X_all, y_all, None


def load_scaler():
    """兼容接口，逐窗口归一化不需要全局 scaler。"""
    return None


def build_inference_sequence(df: pd.DataFrame, scaler=None) -> np.ndarray:
    """
    为推理构建最新一个序列（逐窗口归一化）。

    Returns:
        X: np.ndarray (1, WINDOW_SIZE, FEATURE_DIM)
    """
    df = compute_raw_features(df)
    available = [c for c in FEATURE_NAMES if c in df.columns]
    feat = df[available].values.astype(np.float32)
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

    if len(feat) < WINDOW_SIZE:
        raise ValueError(f"数据行数不足 {WINDOW_SIZE}，无法构建推理序列")

    window = feat[-WINDOW_SIZE:].copy()
    window = _window_zscore(window)
    return window[np.newaxis, :, :]   # (1, WINDOW_SIZE, FEATURE_DIM)
