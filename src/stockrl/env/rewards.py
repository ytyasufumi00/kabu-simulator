from __future__ import annotations

import math
from abc import ABC, abstractmethod


class RewardStrategy(ABC):
    @abstractmethod
    def compute(self, equity_prev: float, equity_curr: float) -> float:
        raise NotImplementedError


class LogReturnReward(RewardStrategy):
    """ポートフォリオ価値のログリターン。

    生の価値差分より外れ値の影響を受けにくく、銘柄間のスケール差にも
    左右されないため報酬ハッキングが起きにくい（v1のデフォルト）。
    """

    def compute(self, equity_prev: float, equity_curr: float) -> float:
        if equity_prev <= 0 or equity_curr <= 0:
            return -1.0
        return math.log(equity_curr / equity_prev)


class RiskAdjustedReward(RewardStrategy):
    """リターンからボラティリティペナルティを引く拡張案（Phase 1.5、v1未使用）。"""

    def __init__(self, volatility_penalty: float = 0.1):
        self._lambda = volatility_penalty
        self._returns: list[float] = []

    def compute(self, equity_prev: float, equity_curr: float) -> float:
        if equity_prev <= 0 or equity_curr <= 0:
            return -1.0
        log_return = math.log(equity_curr / equity_prev)
        self._returns.append(log_return)
        window = self._returns[-20:]
        if len(window) < 2:
            return log_return
        mean = sum(window) / len(window)
        variance = sum((r - mean) ** 2 for r in window) / len(window)
        return log_return - self._lambda * math.sqrt(variance)
