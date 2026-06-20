from __future__ import annotations

import pandas as pd

from .config import Settings
from .data.loader import DataLoader
from .data.yfinance_source import YFinanceSource
from .features.pipeline import compute_features_clean
from .splits import Split, fixed_split


def build_data_loader(settings: Settings) -> DataLoader:
    if settings.data_source != "yfinance":
        raise NotImplementedError(
            f"data_source={settings.data_source} は未対応です（yfinanceのみ対応）"
        )
    return DataLoader(YFinanceSource(), settings.data_cache_dir)


def load_ticker_features(ticker: str, settings: Settings) -> pd.DataFrame:
    loader = build_data_loader(settings)
    date_end = settings.date_end or pd.Timestamp.today().date()
    raw = loader.get(ticker, settings.date_start, date_end)
    if raw.empty:
        raise RuntimeError(f"{ticker} のデータが取得できませんでした")
    return compute_features_clean(raw)


def load_ticker_split(ticker: str, settings: Settings) -> Split:
    features_df = load_ticker_features(ticker, settings)
    return fixed_split(
        features_df,
        train_start=settings.split.train_start,
        train_end=settings.split.train_end,
        test_start=settings.split.test_start,
        test_end=settings.split.test_end,
    )
