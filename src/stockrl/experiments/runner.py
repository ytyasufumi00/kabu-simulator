from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

from ..config import EnvConfig, PPOConfig
from ..env.rewards import LogReturnReward, RewardStrategy, RiskAdjustedReward
from ..evaluation.backtest import buy_and_hold_curve, run_backtest
from ..evaluation.report import save_run_report
from ..splits import Split
from ..training.train_ppo import make_env
from .config import ExperimentVariant

REWARD_FACTORIES = {
    "log_return": LogReturnReward,
    "risk_adjusted": RiskAdjustedReward,
}


def build_reward_strategy(name: str) -> RewardStrategy:
    try:
        factory = REWARD_FACTORIES[name]
    except KeyError:
        raise ValueError(
            f"未知のreward_strategy: {name}（選択肢: {list(REWARD_FACTORIES)}）"
        ) from None
    return factory()


def resolve_ppo_config(base: PPOConfig, overrides: dict, timesteps: int) -> PPOConfig:
    merged = dataclasses.replace(base, timesteps_per_iteration=timesteps, **overrides)
    return merged


@dataclass
class FoldResult:
    fold_index: int
    metrics: dict


@dataclass
class VariantResult:
    variant_name: str
    fold_results: list[FoldResult]

    @property
    def mean_sharpe(self) -> float:
        if not self.fold_results:
            return float("-inf")
        return sum(f.metrics["sharpe_ratio"] for f in self.fold_results) / len(
            self.fold_results
        )


def run_variant(
    ticker: str,
    variant: ExperimentVariant,
    folds: list[Split],
    env_cfg: EnvConfig,
    base_ppo_cfg: PPOConfig,
    experiment_timesteps: int,
    runs_dir: Path,
    settings_path: Path,
) -> VariantResult:
    """1つのvariantを全foldで評価する。各foldはゼロから学習する（fold間の継続学習はしない）。"""
    ppo_cfg = resolve_ppo_config(base_ppo_cfg, variant.ppo_overrides, experiment_timesteps)
    reward_strategy_name = variant.reward_strategy

    fold_results: list[FoldResult] = []
    for i, split in enumerate(folds):
        train_env = make_env(
            split.train, env_cfg, reward_strategy=build_reward_strategy(reward_strategy_name)
        )
        test_env = make_env(
            split.test, env_cfg, reward_strategy=build_reward_strategy(reward_strategy_name)
        )

        variant_dir = runs_dir / ticker / "experiments" / variant.name / f"fold{i}"
        variant_dir.mkdir(parents=True, exist_ok=True)
        best_model_dir = variant_dir / "best_model"

        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=ppo_cfg.learning_rate,
            n_steps=ppo_cfg.n_steps,
            batch_size=ppo_cfg.batch_size,
            gamma=ppo_cfg.gamma,
            verbose=0,
        )

        eval_callback = EvalCallback(
            test_env,
            best_model_save_path=str(best_model_dir),
            eval_freq=max(ppo_cfg.n_steps, 1000),
            n_eval_episodes=1,
            deterministic=True,
            verbose=0,
        )
        model.learn(total_timesteps=ppo_cfg.timesteps_per_iteration, callback=eval_callback)

        best_model_path = best_model_dir / "best_model.zip"
        eval_model = PPO.load(str(best_model_path)) if best_model_path.exists() else model

        eval_env = make_env(
            split.test, env_cfg, reward_strategy=build_reward_strategy(reward_strategy_name)
        )
        result = run_backtest(eval_env, model=eval_model)
        benchmark_curve = buy_and_hold_curve(
            split.test["close"].to_numpy()[env_cfg.window_size :], env_cfg.initial_cash
        )

        metrics = save_run_report(
            run_dir=variant_dir,
            ticker=ticker,
            result=result,
            benchmark_curve=benchmark_curve,
            settings_path=settings_path,
            model_save_fn=lambda path: eval_model.save(str(path)),
        )
        fold_results.append(FoldResult(fold_index=i, metrics=metrics))

    return VariantResult(variant_name=variant.name, fold_results=fold_results)
