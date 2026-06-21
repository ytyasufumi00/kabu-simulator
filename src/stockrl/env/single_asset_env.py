from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from ..features.pipeline import FEATURE_COLUMNS
from .portfolio import Portfolio
from .rewards import LogReturnReward, RewardStrategy

HOLD, BUY, SELL = 0, 1, 2


def build_observation(
    window_features: np.ndarray, portfolio: Portfolio, price: float, initial_cash: float
) -> np.ndarray:
    """直近window分の特徴量とポートフォリオ状態から観測ベクトルを作る。

    学習済みモデルが前提とする観測フォーマットと完全に一致させる必要があるため、
    env内部（_obs）とフォワードテストの両方からこの関数を呼ぶ。
    """
    window = window_features.flatten()
    equity = portfolio.equity(price)
    cash_ratio = portfolio.cash / equity if equity > 0 else 0.0
    has_position = 1.0 if portfolio.shares_held > 0 else 0.0
    unrealized_pnl_ratio = equity / initial_cash - 1.0 if initial_cash > 0 else 0.0
    portfolio_obs = np.array(
        [cash_ratio, has_position, unrealized_pnl_ratio], dtype=np.float32
    )
    return np.concatenate([window, portfolio_obs]).astype(np.float32)


class SingleAssetTradingEnv(gym.Env):
    """1銘柄に対する仮想取引環境（全部買い/全部売り/ホールドの3択）。

    観測 = 直近 window_size 日分の特徴量（フラット化） + ポートフォリオ状態
    （cashの割合、ポジション有無、未実現損益率）。
    執行価格は同バーの close を使用（日次バーでの単純化。README に明記）。
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 10,
        initial_cash: float = 1_000_000.0,
        commission_pct: float = 0.001,
        slippage_bps: float = 0.0,
        reward_strategy: RewardStrategy | None = None,
    ):
        super().__init__()
        missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"df に特徴量列が不足しています: {missing}")

        self._df = df.reset_index(drop=True)
        self._close = self._df["close"].to_numpy(dtype=np.float64)
        self._features = self._df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        self._window_size = window_size
        self._initial_cash = initial_cash
        self._commission_pct = commission_pct
        self._slippage_bps = slippage_bps
        self._reward_strategy = reward_strategy or LogReturnReward()

        n_feature_obs = window_size * len(FEATURE_COLUMNS)
        n_portfolio_obs = 3
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_feature_obs + n_portfolio_obs,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)  # HOLD, BUY, SELL

        self._portfolio: Portfolio | None = None
        self._step_idx: int = 0
        self._max_step_idx = len(self._df) - 1

        if self._max_step_idx < window_size + 1:
            raise ValueError("データ件数が window_size に対して少なすぎます")

    @property
    def portfolio(self) -> Portfolio:
        return self._portfolio

    def _obs(self) -> np.ndarray:
        window = self._features[self._step_idx - self._window_size : self._step_idx]
        price = self._close[self._step_idx]
        return build_observation(window, self._portfolio, price, self._initial_cash)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._portfolio = Portfolio(
            initial_cash=self._initial_cash,
            commission_pct=self._commission_pct,
            slippage_bps=self._slippage_bps,
        )
        self._step_idx = self._window_size
        self._portfolio.record_equity(self._close[self._step_idx])
        return self._obs(), {}

    def step(self, action: int):
        price = self._close[self._step_idx]
        equity_prev = self._portfolio.equity(price)

        if action == BUY:
            self._portfolio.buy_all(self._step_idx, price)
        elif action == SELL:
            self._portfolio.sell_all(self._step_idx, price)

        self._step_idx += 1
        terminated = self._step_idx >= self._max_step_idx
        next_price = self._close[self._step_idx]
        equity_curr = self._portfolio.record_equity(next_price)

        reward = self._reward_strategy.compute(equity_prev, equity_curr)
        obs = self._obs()
        info = {
            "equity": equity_curr,
            "cash": self._portfolio.cash,
            "shares_held": self._portfolio.shares_held,
        }
        return obs, reward, terminated, False, info
