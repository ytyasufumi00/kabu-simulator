from __future__ import annotations

import pandas as pd

"""すべての関数は時点 t の値を計算する際、t以前のデータのみを参照する
（`.rolling()` / `.ewm()` は後方窓のみで先読みなし）。

`tests/test_indicators.py` の truncation-invariance テストでこの性質を検証している。
"""


def returns(close: pd.Series, periods: int = 1) -> pd.Series:
    return close.pct_change(periods=periods)


def sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window=window).mean()


def sma_ratio(close: pd.Series, window: int) -> pd.Series:
    """close / SMA(window) — 生の価格より正規化された、銘柄間で比較可能なトレンド指標。"""
    return close / sma(close, window)


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "macd_signal": signal_line, "macd_hist": histogram}
    )


def volume_change_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
    """直近window日の移動平均出来高に対する比率（生の出来高は銘柄間でスケールが違いすぎるため）。"""
    return volume / volume.rolling(window=window).mean()


def rolling_volatility(close: pd.Series, window: int = 20) -> pd.Series:
    return close.pct_change().rolling(window=window).std()
