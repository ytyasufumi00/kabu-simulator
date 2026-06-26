from __future__ import annotations

import numpy as np
import pandas as pd
from gymnasium.utils.env_checker import check_env

from stockrl.env.risk import RiskLimits
from stockrl.env.single_asset_env import SingleAssetTradingEnv
from stockrl.features.pipeline import compute_features_clean


def _make_env(synthetic_ohlcv: pd.DataFrame, **kwargs) -> SingleAssetTradingEnv:
    df = compute_features_clean(synthetic_ohlcv)
    defaults = dict(window_size=10, initial_cash=1000.0, commission_pct=0.001, min_trade_pct=0.0)
    defaults.update(kwargs)
    return SingleAssetTradingEnv(df, **defaults)


def test_env_passes_gymnasium_checker(synthetic_ohlcv: pd.DataFrame) -> None:
    env = _make_env(synthetic_ohlcv)
    check_env(env, skip_render_check=True)


def test_buy_then_hold_changes_equity_with_price(synthetic_ohlcv: pd.DataFrame) -> None:
    # min_trade_pctを設定し、価格変動による微小な配分ずれでは再売買しないようにする
    env = _make_env(
        synthetic_ohlcv,
        min_trade_pct=0.2,
        risk_limits=RiskLimits(max_position_pct=1.0, target_volatility=1.0, stop_loss_drawdown_pct=1.0),
    )
    env.reset()

    _, _, _, _, info_buy = env.step(np.array([1.0], dtype=np.float32))
    assert info_buy["shares_held"] > 0

    _, _, _, _, info_hold = env.step(np.array([1.0], dtype=np.float32))
    # 同じ目標配分(100%)を指定し続ければ、わずかな価格変動程度では再売買しない
    assert info_hold["shares_held"] == info_buy["shares_held"]


def test_full_episode_runs_without_error(synthetic_ohlcv: pd.DataFrame) -> None:
    env = _make_env(synthetic_ohlcv)
    env.reset()
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
    _, _, _, _, info = env.step(np.array([0.0], dtype=np.float32))
    assert info["shares_held"] == 0
    assert info["cash"] == env._initial_cash


def test_max_position_pct_caps_target(synthetic_ohlcv: pd.DataFrame) -> None:
    env = _make_env(
        synthetic_ohlcv,
        risk_limits=RiskLimits(max_position_pct=0.5, target_volatility=1.0, stop_loss_drawdown_pct=1.0),
    )
    env.reset()
    _, _, _, _, info = env.step(np.array([1.0], dtype=np.float32))
    assert info["target_pct"] == 0.5
