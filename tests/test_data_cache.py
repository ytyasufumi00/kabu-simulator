from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from stockrl.data.base import PriceDataSource
from stockrl.data.loader import DataLoader


class _FakeSource(PriceDataSource):
    name = "fake"

    def __init__(self):
        self.calls = 0

    def fetch_ohlcv(self, ticker, start, end, interval="1d"):
        self.calls += 1
        dates = pd.date_range(start, end, freq="D")
        return pd.DataFrame(
            {
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1000.0,
            },
            index=dates,
        )


def test_loader_caches_and_avoids_refetch(tmp_path: Path) -> None:
    source = _FakeSource()
    loader = DataLoader(source, tmp_path)

    df1 = loader.get("TEST", date(2020, 1, 1), date(2020, 1, 10))
    assert source.calls == 1
    assert not df1.empty

    df2 = loader.get("TEST", date(2020, 1, 1), date(2020, 1, 10))
    assert source.calls == 1  # キャッシュヒットにより再取得されない
    # parquet往復でindexのdatetime64分解能(s/ms)やfreq属性が変わることがあるため値のみ比較
    pd.testing.assert_frame_equal(df1, df2, check_index_type=False, check_freq=False)


def test_loader_refetches_when_range_not_covered(tmp_path: Path) -> None:
    source = _FakeSource()
    loader = DataLoader(source, tmp_path)

    loader.get("TEST", date(2020, 1, 1), date(2020, 1, 10))
    assert source.calls == 1

    loader.get("TEST", date(2020, 1, 1), date(2020, 2, 1))
    assert source.calls == 2
