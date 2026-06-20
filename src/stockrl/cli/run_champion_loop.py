from __future__ import annotations

import argparse

from ..config import SETTINGS_PATH, load_settings
from ..dataset import load_ticker_features
from ..experiments.champion_loop import run_champion_loop
from ..experiments.config import load_experiments_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="walk-forwardで複数variantを評価し、championを自動更新する"
    )
    parser.add_argument("--ticker", required=True)
    args = parser.parse_args()

    settings = load_settings()
    experiments_cfg = load_experiments_config()
    features_df = load_ticker_features(args.ticker, settings)

    decision = run_champion_loop(
        ticker=args.ticker,
        features_df=features_df,
        settings=settings,
        experiments_cfg=experiments_cfg,
        settings_path=SETTINGS_PATH,
    )

    print(f"ticker: {args.ticker}")
    print(f"promoted: {decision.promote}")
    print(f"reason: {decision.reason}")


if __name__ == "__main__":
    main()
