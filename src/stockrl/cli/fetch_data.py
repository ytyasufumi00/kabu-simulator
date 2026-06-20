from __future__ import annotations

import pandas as pd

from ..config import load_settings
from ..dataset import build_data_loader


def main() -> None:
    settings = load_settings()
    loader = build_data_loader(settings)
    date_end = settings.date_end or pd.Timestamp.today().date()

    for ticker in settings.tickers:
        df = loader.get(ticker, settings.date_start, date_end)
        print(f"{ticker}: {len(df)} rows, {df.index.min()} -> {df.index.max()}")


if __name__ == "__main__":
    main()
