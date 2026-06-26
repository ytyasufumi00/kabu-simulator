from __future__ import annotations

import shutil
from datetime import date
from pathlib import Path

from ..config import load_settings
from ..forward_test.aggregate import build_combined_csv
from ..forward_test.config import load_forward_test_config
from ..forward_test.runner import ensure_forward_champion, run_daily_step
from ..forward_test.state import daily_log_path, load_state, save_state


def main() -> None:
    settings = load_settings()
    ft_cfg = load_forward_test_config()
    today = date.today()

    for ticker in ft_cfg.tickers:
        print(f"=== {ticker} ===")
        refreshed = ensure_forward_champion(ticker, settings.runs_dir, today)
        if refreshed:
            print("  forward_championを更新しました")

        state = load_state(ticker, settings.runs_dir, ft_cfg.per_ticker_capital)
        new_state, result = run_daily_step(
            ticker=ticker,
            settings=settings,
            window_size=settings.env.window_size,
            per_ticker_capital=ft_cfg.per_ticker_capital,
            state=state,
        )
        save_state(ticker, settings.runs_dir, new_state)

        if result is None:
            print("  スキップ（forward_champion未作成、または本日分は処理済み）")
        else:
            print(
                f"  {result['date']}: target_pct={result['target_pct']:.1%} "
                f"equity={result['equity']:.0f}"
            )

    ticker_capitals = {t: ft_cfg.per_ticker_capital for t in ft_cfg.tickers}
    dashboard_data_dir = settings.runs_dir.parent / "dashboard" / "data"
    output_path = dashboard_data_dir / "_forward_test" / "combined.csv"
    combined = build_combined_csv(settings.runs_dir, ticker_capitals, output_path)
    if not combined.empty:
        latest = combined.iloc[-1]
        print(
            f"\n合計: 投資総額={latest['total_contributed']:.0f}円 "
            f"評価額={latest['total_equity']:.0f}円"
        )

    _snapshot_forward_test_data(settings.runs_dir, dashboard_data_dir, ft_cfg.tickers)


def _snapshot_forward_test_data(
    runs_dir: Path, dashboard_data_dir: Path, tickers: list[str]
) -> None:
    """銘柄別のフォワードテスト結果（日次ログ・champion更新履歴）をダッシュボード用にコピーする。"""
    for ticker in tickers:
        out_dir = dashboard_data_dir / ticker / "forward_test"
        out_dir.mkdir(parents=True, exist_ok=True)

        log_src = daily_log_path(ticker, runs_dir)
        if log_src.exists():
            shutil.copy(log_src, out_dir / "daily_log.csv")

        history_src = runs_dir / ticker / "forward_champion_history.csv"
        if history_src.exists():
            shutil.copy(history_src, out_dir / "forward_champion_history.csv")


if __name__ == "__main__":
    main()
