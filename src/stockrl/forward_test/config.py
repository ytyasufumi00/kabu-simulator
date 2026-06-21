from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from ..config import PROJECT_ROOT

FORWARD_TEST_PATH = PROJECT_ROOT / "config" / "forward_test.yaml"


@dataclass(frozen=True)
class ForwardTestConfig:
    total_capital: float
    per_ticker_capital: float
    refresh_day_of_month: int
    tickers: list[str]


def load_forward_test_config(path: Path = FORWARD_TEST_PATH) -> ForwardTestConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return ForwardTestConfig(
        total_capital=float(raw["total_capital"]),
        per_ticker_capital=float(raw["per_ticker_capital"]),
        refresh_day_of_month=int(raw["refresh_day_of_month"]),
        tickers=list(raw["tickers"]),
    )
