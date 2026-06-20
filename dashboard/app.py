from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).parent / "data"

TICKER_NAMES = {
    "7203.T": "トヨタ自動車",
    "6758.T": "ソニーグループ",
    "9984.T": "ソフトバンクグループ",
    "8306.T": "三菱UFJフィナンシャル・グループ",
    "9432.T": "日本電信電話",
}

METRIC_LABELS = {
    "cumulative_return": "累積リターン",
    "cagr": "CAGR（年率成長率）",
    "sharpe_ratio": "シャープレシオ",
    "max_drawdown": "最大ドローダウン",
    "win_rate": "勝率",
}


def load_ticker_data(ticker_dir: Path) -> tuple[dict, pd.DataFrame | None]:
    metrics_path = ticker_dir / "metrics.json"
    history_path = ticker_dir / "history.csv"

    metrics = {}
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)

    history = pd.read_csv(history_path) if history_path.exists() else None
    return metrics, history


def build_commentary(metrics: dict, history: pd.DataFrame | None) -> list[str]:
    notes = []

    cum_return = metrics.get("cumulative_return")
    benchmark_return = metrics.get("benchmark_cumulative_return")
    if cum_return is not None and benchmark_return is not None:
        if cum_return > benchmark_return:
            notes.append(
                f"buy&holdの累積リターン（{benchmark_return:+.1%}）を上回っています"
                f"（エージェント: {cum_return:+.1%}）。"
            )
        else:
            notes.append(
                f"buy&holdの累積リターン（{benchmark_return:+.1%}）に届いていません"
                f"（エージェント: {cum_return:+.1%}）。"
            )

    sharpe = metrics.get("sharpe_ratio")
    if sharpe is not None:
        if sharpe > 1.0:
            notes.append(f"シャープレシオは{sharpe:.2f}と良好な水準です。")
        elif sharpe > 0:
            notes.append(f"シャープレシオは{sharpe:.2f}で、リスクに対するリターンはまだ平凡です。")
        else:
            notes.append(f"シャープレシオは{sharpe:.2f}とマイナスで、リスクに見合うリターンが出ていません。")

    max_dd = metrics.get("max_drawdown")
    if max_dd is not None and max_dd < -0.3:
        notes.append(f"最大ドローダウンが{max_dd:.1%}と大きく、実用化前にリスク管理の見直しが必要です。")

    if history is not None and len(history) >= 2:
        first_sharpe = history["sharpe_ratio"].iloc[0]
        last_sharpe = history["sharpe_ratio"].iloc[-1]
        if last_sharpe > first_sharpe:
            notes.append(
                f"成長ループの{len(history)}イテレーションでシャープレシオは"
                f"{first_sharpe:.2f} → {last_sharpe:.2f}に改善しています。"
            )
        else:
            notes.append(
                f"成長ループの{len(history)}イテレーションではシャープレシオの改善は見られません"
                f"（{first_sharpe:.2f} → {last_sharpe:.2f}）。"
            )
    else:
        notes.append("学習イテレーションがまだ少なく、傾向を判断するには時期尚早です。")

    notes.append("これは仮想取引のスナップショットであり、実運用の判断材料としては参考程度に留めてください。")
    return notes


def render_ticker_section(ticker: str, ticker_dir: Path) -> None:
    display_name = TICKER_NAMES.get(ticker, ticker)
    st.header(f"{display_name}（{ticker}）")

    metrics, history = load_ticker_data(ticker_dir)
    if not metrics:
        st.info("この銘柄の学習結果はまだありません。")
        return

    image_path = ticker_dir / "equity_curve.png"
    cols = st.columns([2, 1])
    with cols[0]:
        if image_path.exists():
            st.image(str(image_path), caption="エージェント vs buy&hold（テスト期間）")

    with cols[1]:
        table_rows = {
            METRIC_LABELS.get(k, k): v
            for k, v in metrics.items()
            if k in METRIC_LABELS
        }
        st.table(pd.Series(table_rows, name="値").map(lambda v: f"{v:.4f}"))

    st.subheader("寸評")
    for note in build_commentary(metrics, history):
        st.write(f"- {note}")

    if history is not None and len(history) >= 2:
        st.subheader("成長ループの推移")
        st.line_chart(history.set_index("iteration")[["sharpe_ratio", "cumulative_return"]])

    st.divider()


def main() -> None:
    st.set_page_config(page_title="株式予想システム - 結果ダッシュボード", layout="wide")
    st.title("株式予想システム — 仮想取引結果ダッシュボード")
    st.caption(
        "東証銘柄を対象に、PPOエージェントが仮想取引を通じて売買アルゴリズムを学習した結果のスナップショットです。"
        "学習が完了するたびに更新される自動更新ではなく、手動でスナップショットを取得した時点の結果です。"
    )

    if not DATA_DIR.exists():
        st.warning("データがまだありません。")
        return

    ticker_dirs = sorted(d for d in DATA_DIR.iterdir() if d.is_dir())
    if not ticker_dirs:
        st.warning("データがまだありません。")
        return

    for ticker_dir in ticker_dirs:
        render_ticker_section(ticker_dir.name, ticker_dir)


if __name__ == "__main__":
    main()
