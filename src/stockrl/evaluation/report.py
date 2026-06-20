from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .backtest import BacktestResult
from .metrics import compute_all_metrics


def save_run_report(
    run_dir: Path,
    ticker: str,
    result: BacktestResult,
    benchmark_curve: pd.Series,
    settings_path: Path,
    model_save_fn=None,
) -> dict[str, float]:
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics = compute_all_metrics(result.equity_curve, result.trade_pnls)
    benchmark_metrics = compute_all_metrics(benchmark_curve)
    metrics["benchmark_cumulative_return"] = benchmark_metrics["cumulative_return"]
    metrics["benchmark_sharpe_ratio"] = benchmark_metrics["sharpe_ratio"]

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    trades_df = pd.DataFrame(
        [
            {
                "step": t.step,
                "action": t.action,
                "price": t.price,
                "shares": t.shares,
                "cost": t.cost,
            }
            for t in result.trades
        ]
    )
    trades_df.to_csv(run_dir / "trades.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(result.equity_curve.values, label="agent")
    ax.plot(benchmark_curve.values, label="buy_and_hold")
    ax.set_title(f"{ticker} — equity curve (test period)")
    ax.set_xlabel("step")
    ax.set_ylabel("equity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(run_dir / "equity_curve.png")
    plt.close(fig)

    if settings_path.exists():
        shutil.copy(settings_path, run_dir / "config_snapshot.yaml")

    if model_save_fn is not None:
        model_save_fn(run_dir / "model.zip")

    return metrics


def append_history(history_path: Path, run_id: str, iteration: int, ticker: str, metrics: dict) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    row = {"run_id": run_id, "iteration": iteration, "ticker": ticker, **metrics}
    df_row = pd.DataFrame([row])
    if history_path.exists():
        df_row.to_csv(history_path, mode="a", header=False, index=False)
    else:
        df_row.to_csv(history_path, mode="w", header=True, index=False)
