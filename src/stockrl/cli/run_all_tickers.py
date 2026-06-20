from __future__ import annotations

import argparse

from ..config import SETTINGS_PATH, load_settings
from ..dataset import load_ticker_split
from ..training.growth_loop import run_growth_loop


def main() -> None:
    parser = argparse.ArgumentParser(
        description="settings.yaml の全銘柄について成長ループを独立実行する（銘柄ごとに別モデル）"
    )
    parser.add_argument("--iterations", type=int, default=None)
    args = parser.parse_args()

    settings = load_settings()
    for ticker in settings.tickers:
        print(f"=== {ticker} ===")
        split = load_ticker_split(ticker, settings)
        results = run_growth_loop(
            ticker=ticker,
            train_df=split.train,
            test_df=split.test,
            settings=settings,
            settings_path=SETTINGS_PATH,
            n_iterations=args.iterations,
        )
        last = results[-1]
        print(
            f"  final: cum_return={last['cumulative_return']:.4f} "
            f"sharpe={last['sharpe_ratio']:.4f} vs_buyhold={last['benchmark_cumulative_return']:.4f}"
        )


if __name__ == "__main__":
    main()
