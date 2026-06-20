from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"

load_dotenv(PROJECT_ROOT / ".env")


def _parse_date(value: str | None) -> date | None:
    if value is None:
        return None
    return date.fromisoformat(value)


@dataclass(frozen=True)
class EnvConfig:
    window_size: int
    initial_cash: float
    commission_pct: float
    slippage_bps: float


@dataclass(frozen=True)
class PPOConfig:
    timesteps_per_iteration: int
    learning_rate: float
    n_steps: int
    batch_size: int
    gamma: float


@dataclass(frozen=True)
class SplitConfig:
    train_start: date
    train_end: date
    test_start: date
    test_end: date | None


@dataclass(frozen=True)
class Settings:
    data_source: str
    tickers: list[str]
    date_start: date
    date_end: date | None
    split: SplitConfig
    env: EnvConfig
    ppo: PPOConfig
    growth_loop_iterations: int
    data_cache_dir: Path
    runs_dir: Path

    @property
    def jquants_refresh_token(self) -> str | None:
        import os

        return os.environ.get("JQUANTS_REFRESH_TOKEN") or None


def load_settings(path: Path = SETTINGS_PATH) -> Settings:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    split_raw = raw["split"]
    env_raw = raw["env"]
    ppo_raw = raw["ppo"]
    paths_raw = raw["paths"]

    return Settings(
        data_source=raw["data_source"],
        tickers=list(raw["tickers"]),
        date_start=_parse_date(raw["date_range"]["start"]),
        date_end=_parse_date(raw["date_range"]["end"]),
        split=SplitConfig(
            train_start=_parse_date(split_raw["train_start"]),
            train_end=_parse_date(split_raw["train_end"]),
            test_start=_parse_date(split_raw["test_start"]),
            test_end=_parse_date(split_raw["test_end"]),
        ),
        env=EnvConfig(
            window_size=env_raw["window_size"],
            initial_cash=float(env_raw["initial_cash"]),
            commission_pct=float(env_raw["commission_pct"]),
            slippage_bps=float(env_raw["slippage_bps"]),
        ),
        ppo=PPOConfig(
            timesteps_per_iteration=ppo_raw["timesteps_per_iteration"],
            learning_rate=float(ppo_raw["learning_rate"]),
            n_steps=ppo_raw["n_steps"],
            batch_size=ppo_raw["batch_size"],
            gamma=float(ppo_raw["gamma"]),
        ),
        growth_loop_iterations=raw["growth_loop"]["n_iterations"],
        data_cache_dir=PROJECT_ROOT / paths_raw["data_cache"],
        runs_dir=PROJECT_ROOT / paths_raw["runs"],
    )
