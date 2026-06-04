# 特征构建：对原始OHLCV数据做时间窗口整理，不引入任何人工指标

import os
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import WINDOW_SIZE, LABEL_HORIZON, LABEL_THRESHOLD, MODEL_SAVE_DIR

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "pct_change",        # 涨跌幅（日收益率）
    "turnover",          # 换手率
    "volume_ratio",      # 量比
    "volume_norm",       # 成交量相对20日均值比值
    "main_inflow_ratio", # 主力净流入占比
    "amplitude",         # 振幅
    "open_change",       # 开盘涨幅
    "close_strength",    # 收盘强度
]


def compute_raw_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    基于 OHLCV 原始数据计算8个基础特征列，不使用任何人工技术指标。

    Args:
        df: 包含 open/close/high/low/volume/amount/pct_change/turnover 的 DataFrame

    Returns:
        添加了特征列的 DataFrame
    """
    df = df.copy().reset_index(drop=True)

    for col in ["open", "close", "high", "low", "volume", "amount", "pct_change", "turnover"]:
        if col not in df.columns:
            df[col] = 0.0

    if "volume_ratio" not in df.columns:
        df["volume_ratio"] = 1.0
    if "main_net_inflow" not in df.columns:
        df["main_net_inflow"] = 0.0

    for col in ["open", "close", "high", "low", "volume", "amount", "pct_change", "turnover", "volume_ratio", "main_net_inflow"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # 成交量20日均值归一化
    vol_ma20 = df["volume"].rolling(20, min_periods=1).mean()
    df["volume_norm"] = df["volume"] / (vol_ma20 + 1e-9)

    # 主力净流入占比
    df["main_inflow_ratio"] = df["main_net_inflow"] / (df["amount"].abs() + 1e-9)

    # 振幅：(最高-最低) / 昨收
    prev_close = df["close"].shift(1).fillna(df["close"])
    df["amplitude"] = (df["high"] - df["low"]) / (prev_close + 1e-9)

    # 开盘涨幅：开盘价/昨收 - 1
    df["open_change"] = df["open"] / (prev_close + 1e-9) - 1

    # 收盘强度：(收盘-最低) / (最高-最低)
    hl_range = df["high"] - df["low"]
    df["close_strength"] = np.where(
        hl_range > 0,
        (df["close"] - df["low"]) / hl_range,
        0.5,
    )

    return df


def build_sequences(df: pd.DataFrame, scaler: MinMaxScaler = None, fit_scaler: bool = False):
    """
    将特征 DataFrame 转换为 LSTM 输入序列和标签。

    Args:
        df: compute_raw_features 处理后的 DataFrame
        scaler: 已有的 MinMaxScaler（推理时传入）
        fit_scaler: 是否在此数据上 fit scaler（训练时为 True）

    Returns:
        X: np.ndarray (N, WINDOW_SIZE, 8)
        y: np.ndarray (N,) 二分类标签
        scaler: MinMaxScaler
        dates: 每个样本对应的日期列表
    """
    df = compute_raw_features(df)
    available = [c for c in FEATURE_NAMES if c in df.columns]
    feat = df[available].values.astype(np.float32)
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

    if fit_scaler or scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(feat)

    feat_scaled = scaler.transform(feat)

    # 标签：未来 LABEL_HORIZON 天内最高价相对当日收盘涨幅 > LABEL_THRESHOLD
    closes = df["close"].values
    highs  = df["high"].values
    n = len(df)
    labels = np.zeros(n, dtype=np.float32)
    for i in range(n - LABEL_HORIZON):
        future_high = highs[i + 1: i + 1 + LABEL_HORIZON].max()
        if closes[i] > 0:
            gain = (future_high - closes[i]) / closes[i]
            labels[i] = 1.0 if gain > LABEL_THRESHOLD else 0.0

    X, y, dates = [], [], []
    for i in range(WINDOW_SIZE, n - LABEL_HORIZON):
        X.append(feat_scaled[i - WINDOW_SIZE:i])
        y.append(labels[i])
        dates.append(df["date"].iloc[i] if "date" in df.columns else i)

    return np.array(X), np.array(y), scaler, dates


def build_all_stocks(stock_data_dict: dict, fit_scaler: bool = True):
    """
    对所有股票构建特征序列并合并。

    Args:
        stock_data_dict: {stock_code: DataFrame}
        fit_scaler: 首次训练时为 True

    Returns:
        X, y, scaler
    """
    all_X, all_y = [], []
    scaler = None

    for code, df in stock_data_dict.items():
        if len(df) < WINDOW_SIZE + LABEL_HORIZON + 10:
            logger.warning(f"[{code}] 数据太少（{len(df)}行），跳过")
            continue
        try:
            X, y, scaler, _ = build_sequences(df, scaler=scaler, fit_scaler=(scaler is None and fit_scaler))
            all_X.append(X)
            all_y.append(y)
        except Exception as e:
            logger.error(f"[{code}] 特征构建失败: {e}")

    if not all_X:
        return np.array([]), np.array([]), scaler

    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)

    scaler_path = os.path.join(MODEL_SAVE_DIR, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler 已保存: {scaler_path}")

    return X_all, y_all, scaler


def load_scaler() -> MinMaxScaler:
    """加载已保存的 scaler。"""
    scaler_path = os.path.join(MODEL_SAVE_DIR, "scaler.joblib")
    if os.path.exists(scaler_path):
        return joblib.load(scaler_path)
    raise FileNotFoundError(f"Scaler 未找到: {scaler_path}，请先运行训练")


def build_inference_sequence(df: pd.DataFrame, scaler: MinMaxScaler) -> np.ndarray:
    """
    为推理构建最新一个序列（不需要标签）。

    Returns:
        X: np.ndarray (1, WINDOW_SIZE, 8)
    """
    df = compute_raw_features(df)
    available = [c for c in FEATURE_NAMES if c in df.columns]
    feat = df[available].values.astype(np.float32)
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

    if len(feat) < WINDOW_SIZE:
        raise ValueError(f"数据行数不足 {WINDOW_SIZE}，无法构建推理序列")

    feat_scaled = scaler.transform(feat)
    x = feat_scaled[-WINDOW_SIZE:]
    return x[np.newaxis, :, :]  # (1, 20, 8)
