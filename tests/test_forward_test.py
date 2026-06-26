from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from stockrl.env.portfolio import Portfolio
from stockrl.env.single_asset_env import build_observation
from stockrl.features.pipeline import FEATURE_COLUMNS, compute_features_clean
from stockrl.forward_test.aggregate import build_combined_csv
from stockrl.forward_test.state import ForwardState, load_state, save_state


def test_state_roundtrip(tmp_path: Path) -> None:
    ticker = "TEST.T"
    state = ForwardState(cash=123456.0, shares_held=10.5, last_date="2026-01-15")
    save_state(ticker, tmp_path, state)

    loaded = load_state(ticker, tmp_path, per_ticker_capital=200_000.0)
    assert loaded == state


def test_load_state_defaults_to_initial_capital_when_missing(tmp_path: Path) -> None:
    loaded = load_state("NEW.T", tmp_path, per_ticker_capital=200_000.0)
    assert loaded.cash == 200_000.0
    assert loaded.shares_held == 0.0
    assert loaded.last_date is None


def test_build_observation_matches_env_obs(synthetic_ohlcv: pd.DataFrame) -> None:
    """forward_testのbuild_observation呼び出しが、env内部の_obs()と完全に一致することを保証する。"""
    from stockrl.env.single_asset_env import SingleAssetTradingEnv

    df = compute_features_clean(synthetic_ohlcv)
    env = SingleAssetTradingEnv(df, window_size=10, initial_cash=1000.0, commission_pct=0.001)
    obs_from_env, _ = env.reset()

    features = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    window = features[env._step_idx - 10 : env._step_idx]
    price = df["close"].to_numpy()[env._step_idx]

    obs_standalone = build_observation(window, env.portfolio, price, initial_cash=1000.0)

    np.testing.assert_allclose(obs_from_env, obs_standalone)


def test_build_observation_reflects_existing_position() -> None:
    portfolio = Portfolio(initial_cash=100_000.0, commission_pct=0.0)
    portfolio.cash = 0.0
    portfolio.shares_held = 50.0

    window = np.zeros((10, len(FEATURE_COLUMNS)), dtype=np.float32)
    obs = build_observation(window, portfolio, price=2000.0, initial_cash=100_000.0)

    # cash_ratio=0, position_ratio=1.0（全額投資済み）, unrealized_pnl_ratio = (50*2000)/100000 - 1 = 0.0
    assert obs[-3] == 0.0
    assert obs[-2] == 1.0
    assert obs[-1] == 0.0


def _write_daily_log(runs_dir: Path, ticker: str, rows: list[dict]) -> None:
    path = runs_dir / ticker / "forward_test" / "daily_log.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_build_combined_csv_sums_active_tickers(tmp_path: Path) -> None:
    _write_daily_log(
        tmp_path,
        "A.T",
        [
            {"date": "2026-01-05", "target_pct": 1.0, "price": 100, "cash": 0, "shares_held": 2, "equity": 200_000},
            {"date": "2026-01-06", "target_pct": 1.0, "price": 105, "cash": 0, "shares_held": 2, "equity": 210_000},
        ],
    )
    # B.Tは1月6日から開始（新規追加銘柄を想定）
    _write_daily_log(
        tmp_path,
        "B.T",
        [
            {"date": "2026-01-06", "target_pct": 1.0, "price": 50, "cash": 0, "shares_held": 4, "equity": 200_000},
        ],
    )

    output_path = tmp_path / "combined.csv"
    combined = build_combined_csv(
        tmp_path, {"A.T": 200_000.0, "B.T": 200_000.0}, output_path
    )

    assert output_path.exists()
    combined = combined.set_index("date")

    row_0105 = combined.loc[pd.Timestamp("2026-01-05")]
    assert row_0105["total_contributed"] == 200_000.0
    assert row_0105["total_equity"] == 200_000.0

    row_0106 = combined.loc[pd.Timestamp("2026-01-06")]
    assert row_0106["total_contributed"] == 400_000.0
    assert row_0106["total_equity"] == 210_000.0 + 200_000.0


def test_build_combined_csv_handles_no_data(tmp_path: Path) -> None:
    output_path = tmp_path / "combined.csv"
    combined = build_combined_csv(tmp_path, {"A.T": 200_000.0}, output_path)
    assert combined.empty
    assert output_path.exists()
