from __future__ import annotations

import shutil
from pathlib import Path

from ..config import load_settings


def _latest_run_dir(ticker_runs_dir: Path) -> Path | None:
    candidates = [
        d
        for d in ticker_runs_dir.iterdir()
        if d.is_dir() and d.name != "tensorboard"
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

        latest_run = _latest_run_dir(ticker_runs_dir)
        if latest_run is None:
            print(f"skip {ticker}: run の結果が見つかりません")
            continue

        out_dir = dashboard_data_dir / ticker
        out_dir.mkdir(parents=True, exist_ok=True)

        for filename in ("equity_curve.png", "metrics.json"):
            src = latest_run / filename
            if src.exists():
                shutil.copy(src, out_dir / filename)

        history_src = ticker_runs_dir / "history.csv"
        if history_src.exists():
            shutil.copy(history_src, out_dir / "history.csv")

        print(f"{ticker}: {latest_run.name} のスナップショットを {out_dir} にコピーしました")


if __name__ == "__main__":
    main()
