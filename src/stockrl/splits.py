from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd


@dataclass(frozen=True)
class Split:
    train: pd.DataFrame
    test: pd.DataFrame


def fixed_split(
    df: pd.DataFrame,
    train_start: date,
    train_end: date,
    test_start: date,
    test_end: date | None,
) -> Split:
    """固定の train/test 分割。時系列順を保持し、シャッフルしない。

    特徴量計算済みの df をそのまま渡すこと
    （指標は後方参照のみなので、特徴量計算後に分割しても先読みは発生しない）。
    """
    if test_end is None:
        test_end = df.index.max().date()

    train_mask = (df.index.date >= train_start) & (df.index.date <= train_end)
    test_mask = (df.index.date >= test_start) & (df.index.date <= test_end)

    train_df = df.loc[train_mask]
    test_df = df.loc[test_mask]

    if not train_df.empty and not test_df.empty:
        assert train_df.index.max() < test_df.index.min(), (
            "train期間とtest期間が重複/逆転している（先読みバイアスの危険）"
        )

    return Split(train=train_df, test=test_df)


def walk_forward_splits(
    df: pd.DataFrame, n_splits: int, train_period_days: int, test_period_days: int
) -> list[Split]:
    """ローリング walk-forward 分割（Phase 1.5以降で利用予定のスタブ実装）。

    train が常に対応する test より過去になるよう、重複なく前進させる。
    """
    splits: list[Split] = []
    dates = df.index.sort_values()
    if dates.empty:
        return splits

    cursor = 0
    for _ in range(n_splits):
        train_start_idx = cursor
        train_end_idx = train_start_idx + train_period_days
        test_end_idx = train_end_idx + test_period_days
        if test_end_idx > len(dates):
            break

        train_df = df.iloc[train_start_idx:train_end_idx]
        test_df = df.iloc[train_end_idx:test_end_idx]
        splits.append(Split(train=train_df, test=test_df))
        cursor += test_period_days

    return splits
