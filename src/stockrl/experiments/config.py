from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from ..config import PROJECT_ROOT

EXPERIMENTS_PATH = PROJECT_ROOT / "config" / "experiments.yaml"


@dataclass(frozen=True)
class WalkForwardConfig:
    n_splits: int
    train_period_days: int
    test_period_days: int


@dataclass(frozen=True)
class PromotionConfig:
    min_fold_win_ratio: float
    min_fold_margin: float


@dataclass(frozen=True)
class ExperimentVariant:
    name: str
    reward_strategy: str  # "log_return" | "risk_adjusted"
    ppo_overrides: dict


@dataclass(frozen=True)
class ExperimentsConfig:
    walk_forward: WalkForwardConfig
    promotion: PromotionConfig
    experiment_timesteps: int
    n_envs: int
    variants: list[ExperimentVariant]


def load_experiments_config(path: Path = EXPERIMENTS_PATH) -> ExperimentsConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    wf_raw = raw["walk_forward"]
    promotion_raw = raw["promotion"]

    variants = [
        ExperimentVariant(
            name=v["name"],
            reward_strategy=v["reward_strategy"],
            ppo_overrides=dict(v.get("ppo") or {}),
        )
        for v in raw["variants"]
    ]

    return ExperimentsConfig(
        walk_forward=WalkForwardConfig(
            n_splits=wf_raw["n_splits"],
            train_period_days=wf_raw["train_period_days"],
            test_period_days=wf_raw["test_period_days"],
        ),
        promotion=PromotionConfig(
            min_fold_win_ratio=float(promotion_raw["min_fold_win_ratio"]),
            min_fold_margin=float(promotion_raw["min_fold_margin"]),
        ),
        experiment_timesteps=raw["experiment_timesteps"],
        n_envs=raw.get("n_envs", 1),
        variants=variants,
    )
