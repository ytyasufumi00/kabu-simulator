from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from ..features.pipeline import FEATURE_COLUMNS
from .portfolio import Portfolio
from .rewards import LogReturnReward, RewardStrategy
from .risk import RiskLimits, apply_risk_overlay

_VOLATILITY_IDX = FEATURE_COLUMNS.index("volatility_20")


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
    position_ratio = (portfolio.shares_held * price) / equity if equity > 0 else 0.0
    unrealized_pnl_ratio = equity / initial_cash - 1.0 if initial_cash > 0 else 0.0
    portfolio_obs = np.array(
        [cash_ratio, position_ratio, unrealized_pnl_ratio], dtype=np.float32
    )
    return np.concatenate([window, portfolio_obs]).astype(np.float32)


class SingleAssetTradingEnv(gym.Env):
    """1銘柄に対する仮想取引環境（目標配分0〜100%を毎ステップ指定する連続行動空間）。

    観測 = 直近 window_size 日分の特徴量（フラット化） + ポートフォリオ状態
    （cashの割合、保有比率、未実現損益率）。
    執行価格は同バーの close を使用（日次バーでの単純化。README に明記）。
    モデルの出力（target_pct）には、最大保有比率・ボラティリティターゲティング・
    ストップロスの安全装置（risk.apply_risk_overlay）を学習時にも適用する
    （学習とフォワードテストで同じ制約下の世界を経験させ、train/serveのズレを防ぐ）。
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 10,
        initial_cash: float = 1_000_000.0,
        commission_pct: float = 0.001,
        slippage_bps: float = 0.0,
        min_trade_pct: float = 0.05,
        risk_limits: RiskLimits | None = None,
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
        self._min_trade_pct = min_trade_pct
        self._risk_limits = risk_limits or RiskLimits()
        self._reward_strategy = reward_strategy or LogReturnReward()

        n_feature_obs = window_size * len(FEATURE_COLUMNS)
        n_portfolio_obs = 3
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_feature_obs + n_portfolio_obs,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        self._portfolio: Portfolio | None = None
        self._step_idx: int = 0
        self._peak_equity: float = initial_cash
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
            min_trade_pct=self._min_trade_pct,
        )
        self._step_idx = self._window_size
        equity0 = self._portfolio.record_equity(self._close[self._step_idx])
        self._peak_equity = equity0
        return self._obs(), {}

    def step(self, action: np.ndarray):
        raw_target_pct = float(np.asarray(action).reshape(-1)[0])
        price = self._close[self._step_idx]
        equity_prev = self._portfolio.equity(price)
        # 前日までに確定した値（当日のcloseに基づく値を使うと先読みになるため）
        volatility = float(self._features[self._step_idx - 1, _VOLATILITY_IDX])

        target_pct = apply_risk_overlay(
            raw_target_pct,
            volatility=volatility,
            peak_equity=self._peak_equity,
            current_equity=equity_prev,
            limits=self._risk_limits,
        )
        self._portfolio.rebalance_to(self._step_idx, target_pct, price)

        self._step_idx += 1
        terminated = self._step_idx >= self._max_step_idx
        next_price = self._close[self._step_idx]
        equity_curr = self._portfolio.record_equity(next_price)
        self._peak_equity = max(self._peak_equity, equity_curr)

        reward = self._reward_strategy.compute(equity_prev, equity_curr)
        obs = self._obs()
        info = {
            "equity": equity_curr,
            "cash": self._portfolio.cash,
            "shares_held": self._portfolio.shares_held,
            "target_pct": target_pct,
        }
        return obs, reward, terminated, False, info
