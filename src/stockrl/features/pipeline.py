from __future__ import annotations

import pandas as pd

from . import indicators as ind

FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "sma_ratio_10",
    "sma_ratio_50",
    "rsi_14",
    "macd_hist",
    "volume_change_ratio",
    "volatility_20",
]

# sma_ratio_50 が安定した値を持つために必要な最小ウォームアップ日数
WARMUP_DAYS = 60


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV DataFrame から特徴量列を計算して付加する。

    df は ``open, high, low, close, volume`` 列を持つこと。
    先頭の NaN（ウォームアップ期間分）は呼び出し側で `dropna()` する。
    """
    close = df["close"]
    volume = df["volume"]
    macd_df = ind.macd(close)

    features = pd.DataFrame(index=df.index)
    features["return_1d"] = ind.returns(close, periods=1)
    features["return_5d"] = ind.returns(close, periods=5)
    features["sma_ratio_10"] = ind.sma_ratio(close, window=10)
    features["sma_ratio_50"] = ind.sma_ratio(close, window=50)
    features["rsi_14"] = ind.rsi(close, window=14) / 100.0
    features["macd_hist"] = macd_df["macd_hist"] / close
    features["volume_change_ratio"] = ind.volume_change_ratio(volume, window=20)
    features["volatility_20"] = ind.rolling_volatility(close, window=20)

    return pd.concat([df, features], axis=1)


def compute_features_clean(df: pd.DataFrame) -> pd.DataFrame:
    """compute_features の結果からウォームアップ期間のNaN行を除いたもの。"""
    return compute_features(df).dropna(subset=FEATURE_COLUMNS)
