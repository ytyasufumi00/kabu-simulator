from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ForwardState:
    cash: float
    shares_held: float
    last_date: str | None  # ISO日付文字列。未実行ならNone
    peak_equity: float = 0.0  # ストップロス判定用の直近ピーク評価額


def state_path(ticker: str, runs_dir: Path) -> Path:
    return runs_dir / ticker / "forward_test" / "state.json"


def daily_log_path(ticker: str, runs_dir: Path) -> Path:
    return runs_dir / ticker / "forward_test" / "daily_log.csv"


def load_state(ticker: str, runs_dir: Path, per_ticker_capital: float) -> ForwardState:
    path = state_path(ticker, runs_dir)
    if not path.exists():
        return ForwardState(
            cash=per_ticker_capital,
            shares_held=0.0,
            last_date=None,
            peak_equity=per_ticker_capital,
        )

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return ForwardState(
        cash=float(raw["cash"]),
        shares_held=float(raw["shares_held"]),
        last_date=raw.get("last_date"),
        # 旧形式のstate.json（peak_equity未保存）からの移行時はper_ticker_capitalで初期化する
        peak_equity=float(raw.get("peak_equity", per_ticker_capital)),
    )


def save_state(ticker: str, runs_dir: Path, state: ForwardState) -> None:
    path = state_path(ticker, runs_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "cash": state.cash,
                "shares_held": state.shares_held,
                "last_date": state.last_date,
                "peak_equity": state.peak_equity,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
