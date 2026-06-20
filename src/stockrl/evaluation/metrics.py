from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def cumulative_return(equity_curve: pd.Series) -> float:
    return float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1.0)


def cagr(equity_curve: pd.Series) -> float:
    n_days = len(equity_curve)
    if n_days < 2:
        return 0.0
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    years = n_days / TRADING_DAYS_PER_YEAR
    if total_return <= 0 or years <= 0:
        return -1.0
    return float(total_return ** (1 / years) - 1.0)


def sharpe_ratio(equity_curve: pd.Series, risk_free_rate: float = 0.0) -> float:
    """日次リターンから年率化シャープレシオを計算（無リスク利率は簡略化のため0と仮定）。"""
    daily_returns = equity_curve.pct_change().dropna()
    if daily_returns.std() == 0 or daily_returns.empty:
        return 0.0
    excess = daily_returns - risk_free_rate / TRADING_DAYS_PER_YEAR
    return float(np.sqrt(TRADING_DAYS_PER_YEAR) * excess.mean() / daily_returns.std())


def max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return float(drawdown.min())


def win_rate(trade_pnls: list[float]) -> float:
    if not trade_pnls:
        return 0.0
    wins = sum(1 for pnl in trade_pnls if pnl > 0)
    return wins / len(trade_pnls)


def compute_all_metrics(
    equity_curve: pd.Series, trade_pnls: list[float] | None = None
) -> dict[str, float]:
    return {
        "cumulative_return": cumulative_return(equity_curve),
        "cagr": cagr(equity_curve),
        "sharpe_ratio": sharpe_ratio(equity_curve),
        "max_drawdown": max_drawdown(equity_curve),
        "win_rate": win_rate(trade_pnls or []),
    }
