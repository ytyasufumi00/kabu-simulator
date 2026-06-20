from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..env.portfolio import Trade
from ..env.single_asset_env import SingleAssetTradingEnv


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    actions: pd.Series
    trades: list[Trade]
    trade_pnls: list[float]


def _round_trip_pnls(trades: list[Trade]) -> list[float]:
    pnls = []
    open_trade: Trade | None = None
    for t in trades:
        if t.action == "buy":
            open_trade = t
        elif t.action == "sell" and open_trade is not None:
            cost_basis = open_trade.shares * open_trade.price + open_trade.cost
            proceeds = t.shares * t.price - t.cost
            pnls.append(proceeds - cost_basis)
            open_trade = None
    return pnls


def run_backtest(env: SingleAssetTradingEnv, model=None) -> BacktestResult:
    """学習済みモデル（SB3 PPO等）でテスト環境を1エピソード実行する。

    model=None の場合はランダムエージェント（健全性確認用）。
    """
    obs, _ = env.reset()
    portfolio = env.unwrapped.portfolio
    equities = [portfolio.equity_history[-1]]
    actions = []
    terminated = False
    while not terminated:
        if model is None:
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
        obs, _reward, terminated, _truncated, info = env.step(action)
        equities.append(info["equity"])
        actions.append(action)

    equity_curve = pd.Series(equities, name="equity")
    actions_series = pd.Series(actions, name="action")
    trades = list(portfolio.trades)
    return BacktestResult(
        equity_curve=equity_curve,
        actions=actions_series,
        trades=trades,
        trade_pnls=_round_trip_pnls(trades),
    )


def buy_and_hold_curve(close_prices: np.ndarray, initial_cash: float) -> pd.Series:
    shares = initial_cash / close_prices[0]
    return pd.Series(shares * close_prices, name="buy_and_hold_equity")
