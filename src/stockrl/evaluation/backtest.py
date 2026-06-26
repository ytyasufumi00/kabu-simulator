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
    """各売却について、その時点までの平均取得コストとの差額を実現損益として記録する。

    部分売買（目標配分への都度リバランス）では、1回の売却に複数回の買い増しが
    対応することがあるため、単純な「直前の買いと対」ではなく加重平均コストで計算する。
    """
    pnls: list[float] = []
    open_shares = 0.0
    open_cost = 0.0  # 現在保有株の取得コスト合計（手数料込み）
    for t in trades:
        if t.action == "buy":
            open_shares += t.shares
            open_cost += t.shares * t.price + t.cost
        elif t.action == "sell" and open_shares > 0:
            avg_cost_per_share = open_cost / open_shares
            sold_cost_basis = avg_cost_per_share * t.shares
            proceeds = t.shares * t.price - t.cost
            pnls.append(proceeds - sold_cost_basis)
            open_shares -= t.shares
            open_cost -= sold_cost_basis
    return pnls


def run_backtest(env: SingleAssetTradingEnv, model=None) -> BacktestResult:
    """学習済みモデル（SB3 PPO等）でテスト環境を1エピソード実行する。

    model=None の場合はランダムエージェント（健全性確認用）。
    行動は目標配分（0.0〜1.0の連続値）。
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
        obs, _reward, terminated, _truncated, info = env.step(action)
        equities.append(info["equity"])
        actions.append(info["target_pct"])

    equity_curve = pd.Series(equities, name="equity")
    actions_series = pd.Series(actions, name="target_pct")
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
