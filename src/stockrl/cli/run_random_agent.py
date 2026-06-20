from __future__ import annotations

import argparse

from ..config import load_settings
from ..dataset import load_ticker_split
from ..evaluation.backtest import run_backtest
from ..evaluation.metrics import compute_all_metrics
from ..training.train_ppo import make_env


def main() -> None:
    parser = argparse.ArgumentParser(description="ランダムエージェントで環境の健全性を確認する")
    parser.add_argument("--ticker", required=True)
    args = parser.parse_args()

    settings = load_settings()
    split = load_ticker_split(args.ticker, settings)
    env = make_env(split.test, settings.env)

    result = run_backtest(env, model=None)
    metrics = compute_all_metrics(result.equity_curve, result.trade_pnls)

    print(f"ticker: {args.ticker}")
    print(f"steps: {len(result.equity_curve)}, trades: {len(result.trades)}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
