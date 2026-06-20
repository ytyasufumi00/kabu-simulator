from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd


def cache_path(cache_dir: Path, source: str, ticker: str, interval: str) -> Path:
    safe_ticker = ticker.replace("/", "_")
    return cache_dir / source / f"{safe_ticker}_{interval}.parquet"


def read_cache(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_parquet(path)


def write_cache(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow")


def covers_range(df: pd.DataFrame, start: date, end: date) -> bool:
    if df.empty:
        return False
    return df.index.min().date() <= start and df.index.max().date() >= end
