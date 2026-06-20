from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from ..config import Settings
from ..splits import walk_forward_splits
from .config import ExperimentsConfig, ExperimentVariant
from .runner import FoldResult, VariantResult, run_variant
from .selector import PromotionDecision, compare_to_champion


def load_champion_state(ticker: str, runs_dir: Path) -> VariantResult | None:
    champion_dir = runs_dir / ticker / "champion"
    fold_metrics_path = champion_dir / "fold_metrics.json"
    variant_path = champion_dir / "variant.json"
    if not fold_metrics_path.exists() or not variant_path.exists():
        return None

    with open(variant_path, encoding="utf-8") as f:
        variant_info = json.load(f)
    with open(fold_metrics_path, encoding="utf-8") as f:
        fold_metrics = json.load(f)

    fold_results = [FoldResult(fold_index=i, metrics=m) for i, m in enumerate(fold_metrics)]
    return VariantResult(variant_name=variant_info["name"], fold_results=fold_results)


def promote_champion(
    ticker: str, variant: ExperimentVariant, result: VariantResult, runs_dir: Path
) -> None:
    """勝利したvariantの最終fold（最新期間）のアーティファクトをchampion/に昇格コピーする。"""
    champion_dir = runs_dir / ticker / "champion"
    champion_dir.mkdir(parents=True, exist_ok=True)

    last_fold_idx = len(result.fold_results) - 1
    last_fold_dir = runs_dir / ticker / "experiments" / variant.name / f"fold{last_fold_idx}"

    for filename in ("model.zip", "equity_curve.png", "metrics.json", "trades.csv"):
        src = last_fold_dir / filename
        if src.exists():
            shutil.copy(src, champion_dir / filename)

    with open(champion_dir / "variant.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "name": variant.name,
                "reward_strategy": variant.reward_strategy,
                "ppo_overrides": variant.ppo_overrides,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(champion_dir / "fold_metrics.json", "w", encoding="utf-8") as f:
        json.dump([fr.metrics for fr in result.fold_results], f, ensure_ascii=False, indent=2)


def append_promotion_log(
    runs_dir: Path, ticker: str, variant_name: str, decision: PromotionDecision
) -> None:
    log_path = runs_dir / ticker / "promotions.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "variant_name": variant_name,
        "win_ratio": decision.win_ratio,
        "margin": decision.margin,
        "promoted": decision.promote,
        "reason": decision.reason,
    }
    df_row = pd.DataFrame([row])
    if log_path.exists():
        df_row.to_csv(log_path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(log_path, mode="w", header=True, index=False)


def run_champion_loop(
    ticker: str,
    features_df: pd.DataFrame,
    settings: Settings,
    experiments_cfg: ExperimentsConfig,
    settings_path: Path,
) -> PromotionDecision:
    """全variantをwalk-forwardで評価し、最良のchallengerをchampionと比較、必要なら自動昇格する。

    判定はすべて決定的なPythonロジック（selector.compare_to_champion）であり、
    Claude等のLLM APIは呼び出さない。
    """
    folds = walk_forward_splits(
        features_df,
        n_splits=experiments_cfg.walk_forward.n_splits,
        train_period_days=experiments_cfg.walk_forward.train_period_days,
        test_period_days=experiments_cfg.walk_forward.test_period_days,
    )
    if not folds:
        raise RuntimeError(
            f"{ticker}: walk-forward分割に十分なデータがありません"
            f"（n_splits={experiments_cfg.walk_forward.n_splits}に対してデータ期間が不足）"
        )

    champion = load_champion_state(ticker, settings.runs_dir)

    variant_results: dict[str, VariantResult] = {}
    for variant in experiments_cfg.variants:
        variant_results[variant.name] = run_variant(
            ticker=ticker,
            variant=variant,
            folds=folds,
            env_cfg=settings.env,
            base_ppo_cfg=settings.ppo,
            experiment_timesteps=experiments_cfg.experiment_timesteps,
            runs_dir=settings.runs_dir,
            settings_path=settings_path,
            n_envs=experiments_cfg.n_envs,
        )

    best_variant_name = max(variant_results, key=lambda name: variant_results[name].mean_sharpe)
    best_result = variant_results[best_variant_name]
    best_variant = next(v for v in experiments_cfg.variants if v.name == best_variant_name)

    decision = compare_to_champion(best_result, champion, experiments_cfg.promotion)
    append_promotion_log(settings.runs_dir, ticker, best_variant_name, decision)

    if decision.promote:
        promote_champion(ticker, best_variant, best_result, settings.runs_dir)

    return decision
