from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from ..config import Settings
from ..evaluation.backtest import buy_and_hold_curve, run_backtest
from ..evaluation.report import append_history, save_run_report
from .train_ppo import make_env, train_one_iteration


def run_growth_loop(
    ticker: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    settings: Settings,
    settings_path: Path,
    n_iterations: int | None = None,
) -> list[dict]:
    """学習→バックテスト→メトリクス記録→保存をN回繰り返す「成長ループ」。

    各イテレーションは同じ train/test ウィンドウのまま学習を継続するため、
    「学習を重ねるほど性能が伸びているか」を runs/{ticker}/history.csv の
    推移で観察できる。これが「仮想取引の経過で実用化を判断する」軸になる。
    """
    n_iterations = n_iterations or settings.growth_loop_iterations
    ticker_runs_dir = settings.runs_dir / ticker.replace("/", "_")
    history_path = ticker_runs_dir / "history.csv"
    model_path = ticker_runs_dir / "model.zip"
    tensorboard_log = settings.runs_dir / "tensorboard"

    benchmark_curve = buy_and_hold_curve(
        test_df["close"].to_numpy()[settings.env.window_size :],
        settings.env.initial_cash,
    )

    results = []
    for i in range(n_iterations):
        model = train_one_iteration(
            train_df=train_df,
            env_cfg=settings.env,
            ppo_cfg=settings.ppo,
            model_path_in=model_path if i > 0 else None,
            tensorboard_log=tensorboard_log,
        )
        model.save(str(model_path))

        test_env = make_env(test_df, settings.env)
        result = run_backtest(test_env, model=model)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{ticker}_{timestamp}_{i:03d}"
        run_dir = ticker_runs_dir / run_id

        metrics = save_run_report(
            run_dir=run_dir,
            ticker=ticker,
            result=result,
            benchmark_curve=benchmark_curve,
            settings_path=settings_path,
        )
        append_history(history_path, run_id, i, ticker, metrics)
        results.append({"run_id": run_id, "iteration": i, **metrics})

    return results
