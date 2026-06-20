from __future__ import annotations

import pandas as pd

from stockrl.splits import walk_forward_splits


def test_walk_forward_folds_are_chronological_and_non_overlapping(
    synthetic_ohlcv: pd.DataFrame,
) -> None:
    splits = walk_forward_splits(
        synthetic_ohlcv, n_splits=3, train_period_days=100, test_period_days=50
    )
    assert len(splits) >= 1

    for split in splits:
        assert not split.train.empty
        assert not split.test.empty
        assert split.train.index.max() < split.test.index.min()

    for earlier, later in zip(splits, splits[1:]):
        assert earlier.test.index.max() < later.test.index.min()
        assert earlier.train.index.min() < later.train.index.min()


def test_walk_forward_stops_when_data_insufficient(synthetic_ohlcv: pd.DataFrame) -> None:
    # synthetic_ohlcvは300日分。極端に大きいfold要求では、要求したn_splitsより少ない数しか返らない
    splits = walk_forward_splits(
        synthetic_ohlcv, n_splits=10, train_period_days=100, test_period_days=50
    )
    assert 0 < len(splits) < 10


def test_walk_forward_returns_empty_list_for_empty_input() -> None:
    empty_df = pd.DataFrame(columns=["close"])
    splits = walk_forward_splits(empty_df, n_splits=3, train_period_days=100, test_period_days=50)
    assert splits == []
