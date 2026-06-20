from __future__ import annotations

from ..config import SETTINGS_PATH, load_settings
from ..dataset import load_ticker_features
from ..experiments.champion_loop import run_champion_loop
from ..experiments.config import load_experiments_config


def main() -> None:
    settings = load_settings()
    experiments_cfg = load_experiments_config()

    for ticker in settings.tickers:
        print(f"=== {ticker} ===")
        features_df = load_ticker_features(ticker, settings)
        decision = run_champion_loop(
            ticker=ticker,
            features_df=features_df,
            settings=settings,
            experiments_cfg=experiments_cfg,
            settings_path=SETTINGS_PATH,
        )
        print(f"  promoted: {decision.promote} ({decision.reason})")


if __name__ == "__main__":
    main()
