from __future__ import annotations

import pandas as pd
import pytest

from stockrl.evaluation.metrics import (
    cumulative_return,
    max_drawdown,
    sharpe_ratio,
    win_rate,
)


def test_cumulative_return_doubles() -> None:
    curve = pd.Series([100.0, 150.0, 200.0])
    assert cumulative_return(curve) == pytest.approx(1.0)


def test_max_drawdown_detects_peak_to_trough() -> None:
    curve = pd.Series([100.0, 200.0, 100.0, 150.0])
    assert max_drawdown(curve) == pytest.approx(-0.5)


def test_sharpe_ratio_zero_for_flat_curve() -> None:
    curve = pd.Series([100.0] * 10)
    assert sharpe_ratio(curve) == 0.0


def test_win_rate_counts_profitable_trades() -> None:
    assert win_rate([10.0, -5.0, 3.0, -1.0]) == pytest.approx(0.5)
    assert win_rate([]) == 0.0
