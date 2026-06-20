from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from .base import PriceDataSource
from .cache import cache_path, covers_range, read_cache, write_cache


class DataLoader:
    """`PriceDataSource` の前段にキャッシュ層を挟んで提供する。

    同じ ticker/interval に対するキャッシュが要求範囲をカバーしていれば
    再ダウンロードせずファイルから読み込む。カバーしていなければソースから
    再取得してキャッシュを上書きする（Phase 1 は差分取得をせず単純化）。
    """

    def __init__(self, source: PriceDataSource, cache_dir: Path):
        self._source = source
        self._cache_dir = cache_dir

    def get(
        self, ticker: str, start: date, end: date, interval: str = "1d"
    ) -> pd.DataFrame:
        path = cache_path(self._cache_dir, self._source.name, ticker, interval)
        cached = read_cache(path)
        if cached is not None and covers_range(cached, start, end):
            mask = (cached.index.date >= start) & (cached.index.date <= end)
            return cached.loc[mask]

        df = self._source.fetch_ohlcv(ticker, start, end, interval)
        write_cache(path, df)
        return df
