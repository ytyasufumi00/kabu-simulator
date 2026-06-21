from __future__ import annotations

import shutil
from pathlib import Path

from ..config import load_settings


_NON_RUN_DIR_NAMES = {"tensorboard", "champion", "experiments"}


def _latest_run_dir(ticker_runs_dir: Path) -> Path | None:
    candidates = [
        d
        for d in ticker_runs_dir.iterdir()
        if d.is_dir() and d.name not in _NON_RUN_DIR_NAMES
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.stat().st_mtime)


def main() -> None:
    settings = load_settings()
    dashboard_data_dir = settings.runs_dir.parent / "dashboard" / "data"

    for ticker in settings.tickers:
        ticker_runs_dir = settings.runs_dir / ticker
        if not ticker_runs_dir.exists():
            print(f"skip {ticker}: runs ディレクトリが存在しません（まだ学習していない）")
            continue

        champion_dir = ticker_runs_dir / "champion"
        if champion_dir.exists() and (champion_dir / "metrics.json").exists():
            source_dir = champion_dir
            source_label = "champion（自動選定された最良アルゴリズム）"
        else:
            source_dir = _latest_run_dir(ticker_runs_dir)
            source_label = source_dir.name if source_dir else None

        if source_dir is None:
            print(f"skip {ticker}: run の結果が見つかりません")
            continue

        out_dir = dashboard_data_dir / ticker
        out_dir.mkdir(parents=True, exist_ok=True)

        for filename in ("equity_curve.png", "metrics.json", "variant.json", "trades.csv"):
            src = source_dir / filename
            if src.exists():
                shutil.copy(src, out_dir / filename)

        history_src = ticker_runs_dir / "history.csv"
        if history_src.exists():
            shutil.copy(history_src, out_dir / "history.csv")

        promotions_src = ticker_runs_dir / "promotions.csv"
        if promotions_src.exists():
            shutil.copy(promotions_src, out_dir / "promotions.csv")

        print(f"{ticker}: {source_label} のスナップショットを {out_dir} にコピーしました")


if __name__ == "__main__":
    main()
