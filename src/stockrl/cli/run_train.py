from __future__ import annotations

import argparse
from datetime import datetime

from ..config import SETTINGS_PATH, load_settings
from ..dataset import load_ticker_split
from ..evaluation.backtest import buy_and_hold_curve, run_backtest
from ..evaluation.report import save_run_report
from ..training.train_ppo import make_env, train_one_iteration


def main() -> None:
    parser = argparse.ArgumentParser(description="1銘柄でPPOを1回学習し評価レポートを生成する")
    parser.add_argument("--ticker", required=True)
    args = parser.parse_args()

    settings = load_settings()
    split = load_ticker_split(args.ticker, settings)

    model = train_one_iteration(
        train_df=split.train,
        env_cfg=settings.env,
        ppo_cfg=settings.ppo,
        model_path_in=None,
        tensorboard_log=settings.runs_dir / "tensorboard",
    )

    test_env = make_env(split.test, settings.env)
    result = run_backtest(test_env, model=model)

    benchmark_curve = buy_and_hold_curve(
        split.test["close"].to_numpy()[settings.env.window_size :],
        settings.env.initial_cash,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{args.ticker}_{timestamp}_single"
    run_dir = settings.runs_dir / args.ticker.replace("/", "_") / run_id

    metrics = save_run_report(
        run_dir=run_dir,
        ticker=args.ticker,
        result=result,
        benchmark_curve=benchmark_curve,
        settings_path=SETTINGS_PATH,
        model_save_fn=lambda path: model.save(str(path)),
    )

    print(f"run saved to: {run_dir}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
