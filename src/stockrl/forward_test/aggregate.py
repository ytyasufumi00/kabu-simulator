from __future__ import annotations

from pathlib import Path

import pandas as pd

from .state import daily_log_path


def build_combined_csv(
    runs_dir: Path, ticker_capitals: dict[str, float], output_path: Path
) -> pd.DataFrame:
    """各銘柄のdaily_log.csvを統合し、日次の投資総額と評価額（全銘柄合計）を計算する。

    投資総額: その日までに開始している銘柄のper_ticker_capitalの合計
    （銘柄追加は新規資金の追加投資として扱い、既存銘柄からの引き出し・
    再配分は行わない簡略化）。
    評価額: その日までの各銘柄の最新equity（前方補完）の合計。
    """
    per_ticker_series: dict[str, pd.Series] = {}
    start_dates: dict[str, pd.Timestamp] = {}

    for ticker, capital in ticker_capitals.items():
        path = daily_log_path(ticker, runs_dir)
        if not path.exists():
            continue
        df = pd.read_csv(path, parse_dates=["date"])
        if df.empty:
            continue
        series = df.set_index("date")["equity"]
        per_ticker_series[ticker] = series
        start_dates[ticker] = series.index.min()

    if not per_ticker_series:
        combined = pd.DataFrame(columns=["date", "total_contributed", "total_equity"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output_path, index=False)
        return combined

    all_dates = sorted(set().union(*(s.index for s in per_ticker_series.values())))
    index = pd.DatetimeIndex(all_dates)

    equity_df = pd.DataFrame(index=index)
    for ticker, series in per_ticker_series.items():
        equity_df[ticker] = series.reindex(index).ffill()

    total_equity = equity_df.sum(axis=1, skipna=True)

    contributed = pd.Series(0.0, index=index)
    for ticker, start in start_dates.items():
        contributed = contributed + (index >= start) * ticker_capitals[ticker]

    combined = pd.DataFrame(
        {
            "date": index,
            "total_contributed": contributed.to_numpy(),
            "total_equity": total_equity.to_numpy(),
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    return combined
