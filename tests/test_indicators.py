from __future__ import annotations

import pandas as pd
import pytest

from stockrl.features.pipeline import FEATURE_COLUMNS, compute_features


def test_truncation_invariance(synthetic_ohlcv: pd.DataFrame) -> None:
    """時点tの特徴量は、t以降のデータを切り詰めても変化しない（先読みバイアスがない）。"""
    full = compute_features(synthetic_ohlcv)

    cutoff = 200
    truncated = compute_features(synthetic_ohlcv.iloc[: cutoff + 1])

    for col in FEATURE_COLUMNS:
        full_value = full[col].iloc[cutoff]
        truncated_value = truncated[col].iloc[cutoff]
        if pd.isna(full_value) and pd.isna(truncated_value):
            continue
        assert full_value == pytest.approx(truncated_value, rel=1e-9, abs=1e-9), (
            f"{col} が将来データに依存している（先読みバイアス）"
        )


def test_compute_features_clean_drops_warmup_nans(synthetic_ohlcv: pd.DataFrame) -> None:
    from stockrl.features.pipeline import compute_features_clean

    clean = compute_features_clean(synthetic_ohlcv)
    assert not clean[FEATURE_COLUMNS].isna().any().any()
    assert len(clean) < len(synthetic_ohlcv)
