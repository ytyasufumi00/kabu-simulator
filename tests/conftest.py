from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    rng = np.random.default_rng(seed=42)
    n = 300
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    drift = np.linspace(100, 150, n)
    noise = rng.normal(0, 2, n).cumsum()
    close = drift + noise
    close = np.clip(close, 10, None)

    df = pd.DataFrame(
        {
            "open": close * 0.99,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": rng.integers(1_000, 10_000, n).astype(float),
        },
        index=dates,
    )
    df.index.name = "date"
    return df
