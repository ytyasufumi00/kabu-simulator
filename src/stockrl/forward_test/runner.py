from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path

import pandas as pd
from gymnasium import spaces
from stable_baselines3 import PPO

from ..config import Settings
from ..dataset import load_ticker_features
from ..env.portfolio import Portfolio
from ..env.risk import RiskLimits, apply_risk_overlay
from ..env.single_asset_env import build_observation
from ..features.pipeline import FEATURE_COLUMNS
from .state import ForwardState, daily_log_path

_VOLATILITY_IDX = FEATURE_COLUMNS.index("volatility_20")


def ensure_forward_champion(ticker: str, runs_dir: Path, today: date) -> bool:
    """championのスナップショットをforward_championにコピーする（月初の初回実行時のみ）。

    championがまだ存在しない銘柄や、今月既にスナップショット済みの場合はFalseを返す。
    """
    champion_dir = runs_dir / ticker / "champion"
    if not (champion_dir / "model.zip").exists():
        return False

    forward_dir = runs_dir / ticker / "forward_champion"
    meta_path = forward_dir / "meta.json"
    current_month = today.strftime("%Y-%m")

    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("snapshot_month") == current_month:
            return False

    forward_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("model.zip", "variant.json", "metrics.json"):
        src = champion_dir / filename
        if src.exists():
            shutil.copy(src, forward_dir / filename)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"snapshot_month": current_month}, f, ensure_ascii=False, indent=2)

    history_path = runs_dir / ticker / "forward_champion_history.csv"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([{"month": current_month, "snapshotted_at": today.isoformat()}])
    if history_path.exists():
        row.to_csv(history_path, mode="a", header=False, index=False)
    else:
        row.to_csv(history_path, mode="w", header=True, index=False)

    return True


def _append_daily_log(
    ticker: str, runs_dir: Path, row: dict
) -> None:
    path = daily_log_path(ticker, runs_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    df_row = pd.DataFrame([row])
    if path.exists():
        df_row.to_csv(path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(path, mode="w", header=True, index=False)


def run_daily_step(
    ticker: str,
    settings: Settings,
    window_size: int,
    per_ticker_capital: float,
    state: ForwardState,
) -> tuple[ForwardState, dict | None]:
    """1銘柄について最新データで1日分のフォワードテストを進める。

    モデルは目標配分（0.0〜1.0）を出力し、最大保有比率・ボラティリティターゲティング・
    ストップロスの安全装置（risk.apply_risk_overlay）を適用したうえで部分リバランスする。
    forward_championが存在しない、または既に最新日まで処理済みの場合は
    (state, None) を返す（呼び出し側はstateをそのまま保存してよい）。
    """
    runs_dir = settings.runs_dir
    forward_model_path = runs_dir / ticker / "forward_champion" / "model.zip"
    if not forward_model_path.exists():
        return state, None

    features_df = load_ticker_features(ticker, settings)
    if len(features_df) < window_size + 1:
        return state, None

    latest_date = features_df.index[-1].date().isoformat()
    if state.last_date is not None and latest_date <= state.last_date:
        return state, None

    window_features = features_df[FEATURE_COLUMNS].to_numpy(dtype="float32")[-window_size:]
    price = float(features_df["close"].iloc[-1])
    # 前日までに確定した値（当日のcloseに基づく値を使うと先読みになるため）
    volatility = float(features_df[FEATURE_COLUMNS[_VOLATILITY_IDX]].iloc[-2])

    portfolio = Portfolio(
        initial_cash=per_ticker_capital,
        commission_pct=settings.env.commission_pct,
        slippage_bps=settings.env.slippage_bps,
        min_trade_pct=settings.env.min_trade_pct,
    )
    portfolio.cash = state.cash
    portfolio.shares_held = state.shares_held

    obs = build_observation(window_features, portfolio, price, per_ticker_capital)

    model = PPO.load(str(forward_model_path))
    raw_action, _ = model.predict(obs, deterministic=True)
    equity_before = portfolio.equity(price)
    peak_equity = max(state.peak_equity, equity_before)

    if isinstance(model.action_space, spaces.Box):
        raw_target_pct = float(raw_action.reshape(-1)[0])
        risk_limits = RiskLimits(
            max_position_pct=settings.env.max_position_pct,
            target_volatility=settings.env.target_volatility,
            stop_loss_drawdown_pct=settings.env.stop_loss_drawdown_pct,
        )
        target_pct = apply_risk_overlay(
            raw_target_pct,
            volatility=volatility,
            peak_equity=peak_equity,
            current_equity=equity_before,
            limits=risk_limits,
        )
        portfolio.rebalance_to(0, target_pct, price)
    else:
        # 旧形式（離散行動、全額買い/全額売り）のchampionがまだ残っている銘柄向けの後方互換パス。
        # 学習時にリスクオーバーレイを適用していないため、ここでも適用しない（train/serve skew防止）。
        action = int(raw_action)
        if action == 1:  # BUY
            portfolio.rebalance_to(0, 1.0, price)
        elif action == 2:  # SELL
            portfolio.rebalance_to(0, 0.0, price)
        target_pct = portfolio.position_pct(price)

    equity = portfolio.equity(price)
    peak_equity = max(peak_equity, equity)

    _append_daily_log(
        ticker,
        runs_dir,
        {
            "date": latest_date,
            "target_pct": target_pct,
            "price": price,
            "cash": portfolio.cash,
            "shares_held": portfolio.shares_held,
            "equity": equity,
        },
    )

    new_state = ForwardState(
        cash=portfolio.cash,
        shares_held=portfolio.shares_held,
        last_date=latest_date,
        peak_equity=peak_equity,
    )
    return new_state, {
        "date": latest_date,
        "target_pct": target_pct,
        "equity": equity,
    }
