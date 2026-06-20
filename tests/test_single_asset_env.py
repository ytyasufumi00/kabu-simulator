from __future__ import annotations

import pandas as pd
from gymnasium.utils.env_checker import check_env

from stockrl.env.single_asset_env import BUY, HOLD, SELL, SingleAssetTradingEnv
from stockrl.features.pipeline import compute_features_clean


def _make_env(synthetic_ohlcv: pd.DataFrame) -> SingleAssetTradingEnv:
    df = compute_features_clean(synthetic_ohlcv)
    return SingleAssetTradingEnv(df, window_size=10, initial_cash=1000.0, commission_pct=0.001)


def test_env_passes_gymnasium_checker(synthetic_ohlcv: pd.DataFrame) -> None:
    env = _make_env(synthetic_ohlcv)
    check_env(env, skip_render_check=True)


def test_buy_then_hold_changes_equity_with_price(synthetic_ohlcv: pd.DataFrame) -> None:
    env = _make_env(synthetic_ohlcv)
    obs, _ = env.reset()

    _, _, terminated, _, info_buy = env.step(BUY)
    equity_after_buy = info_buy["equity"]
    assert info_buy["shares_held"] > 0

    _, _, terminated, _, info_hold = env.step(HOLD)
    # ホールド中も保有株の評価額は価格変動に応じて変わる
    assert info_hold["shares_held"] == info_buy["shares_held"]


def test_full_episode_runs_without_error(synthetic_ohlcv: pd.DataFrame) -> None:
    env = _make_env(synthetic_ohlcv)
    obs, _ = env.reset()
    terminated = False
    steps = 0
    while not terminated:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        assert steps < 10_000  # 無限ループ防止
    assert steps > 0


def test_sell_without_position_is_noop(synthetic_ohlcv: pd.DataFrame) -> None:
    env = _make_env(synthetic_ohlcv)
    env.reset()
    _, _, _, _, info = env.step(SELL)
    assert info["shares_held"] == 0
    assert info["cash"] == env._initial_cash
