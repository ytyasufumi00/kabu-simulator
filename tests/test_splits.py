from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from stockrl.splits import fixed_split


def test_fixed_split_train_precedes_test(synthetic_ohlcv: pd.DataFrame) -> None:
    split = fixed_split(
        synthetic_ohlcv,
        train_start=date(2020, 1, 1),
        train_end=date(2020, 6, 1),
        test_start=date(2020, 6, 2),
        test_end=None,
    )
    assert not split.train.empty
    assert not split.test.empty
    assert split.train.index.max() < split.test.index.min()


def test_fixed_split_rejects_overlap(synthetic_ohlcv: pd.DataFrame) -> None:
    with pytest.raises(AssertionError):
        fixed_split(
            synthetic_ohlcv,
            train_start=date(2020, 1, 1),
            train_end=date(2020, 6, 1),
            test_start=date(2020, 5, 1),  # train_endより前 = 重複
            test_end=None,
        )
