from __future__ import annotations

import argparse

from ..config import SETTINGS_PATH, load_settings
from ..dataset import load_ticker_split
from ..training.growth_loop import run_growth_loop


def main() -> None:
    parser = argparse.ArgumentParser(description="成長ループ（学習→評価→保存のN回繰り返し）を実行する")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--iterations", type=int, default=None)
    args = parser.parse_args()

    settings = load_settings()
    split = load_ticker_split(args.ticker, settings)

    results = run_growth_loop(
        ticker=args.ticker,
        train_df=split.train,
        test_df=split.test,
        settings=settings,
        settings_path=SETTINGS_PATH,
        n_iterations=args.iterations,
    )

    print(f"{len(results)} iterations completed for {args.ticker}")
    for r in results:
        print(
            f"  iter {r['iteration']}: cum_return={r['cumulative_return']:.4f} "
            f"sharpe={r['sharpe_ratio']:.4f} max_dd={r['max_drawdown']:.4f} "
            f"vs_buyhold={r['benchmark_cumulative_return']:.4f}"
        )


if __name__ == "__main__":
    main()
