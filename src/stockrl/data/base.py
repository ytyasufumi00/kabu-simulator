from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import pandas as pd

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


class PriceDataSource(ABC):
    """銘柄のOHLCV日次データを取得するためのインターフェース。

    実装は必ず ``open, high, low, close, volume`` 列を持つ DataFrame を
    日付（tz-naive）でソートされたインデックスとして返すこと。
    """

    name: str

    @abstractmethod
    def fetch_ohlcv(
        self, ticker: str, start: date, end: date, interval: str = "1d"
    ) -> pd.DataFrame:
        raise NotImplementedError
