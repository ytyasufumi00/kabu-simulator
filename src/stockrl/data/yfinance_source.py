from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from .base import OHLCV_COLUMNS, PriceDataSource


class YFinanceSource(PriceDataSource):
    name = "yfinance"

    def fetch_ohlcv(
        self, ticker: str, start: date, end: date, interval: str = "1d"
    ) -> pd.DataFrame:
        # yfinance の end は exclusive なので 1 日加算して指定日を含める
        df = yf.download(
            ticker,
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
        if df.empty:
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.rename(columns=str.lower)[OHLCV_COLUMNS]
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "date"
        return df.sort_index()
