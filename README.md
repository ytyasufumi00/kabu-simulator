# 株式予想システム（Phase 1）

東証銘柄（3〜5銘柄、日次データ）を対象に、強化学習（PPO）エージェントが仮想取引を通じて
売買アルゴリズムを自律的に成長させる基盤。仮想取引の経過（リターン・シャープレシオ・
最大ドローダウン等の推移）から、将来的な実用化可能性を判断するためのツール。

## セットアップ

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .
```

## 使い方

```powershell
# 1. データ取得（config/settings.yamlのtickers分をキャッシュ）
python -m stockrl.cli.fetch_data

# 2. 環境の健全性確認（ランダムエージェント）
python -m stockrl.cli.run_random_agent --ticker 7203.T

# 3. PPOを1回学習して評価レポートを生成
python -m stockrl.cli.run_train --ticker 7203.T

# 4. 成長ループ（学習→評価→保存をN回繰り返す）
python -m stockrl.cli.run_growth_loop --ticker 7203.T --iterations 5

# 5. settings.yamlの全銘柄について成長ループを実行
python -m stockrl.cli.run_all_tickers --iterations 5
```

レポートは `runs/{ticker}/{run_id}/` に保存される（equity曲線・metrics.json・trades.csv・モデル）。
イテレーションをまたいだ推移は `runs/{ticker}/history.csv` に蓄積される。

## 設計上の既知の簡略化（Phase 1）

- 執行価格は意思決定と同じバーの close を使用（実際の約定との差はスリッページとして
  `config/settings.yaml` の `slippage_bps` で将来調整可能）。
- train/testは固定の1分割のみ（walk-forwardは未実装）。単一期間での結果は
  「パイプラインが動くか」の確認であり、「戦略が優れているか」の結論には使えない。
- 1モデル=1銘柄。複数銘柄を1つのRL環境で同時に扱うマルチアセット環境は未実装
  （`src/stockrl/env/multi_asset_env.py` にスタブのみ）。
- データソースはyfinanceのみ実装。J-Quantsはインターフェース・認証スタブのみ
  （`src/stockrl/data/jquants_source.py`）。
- サバイバーシップバイアス: 対象銘柄は現在上場している大型株であり、過去に上場廃止した
  銘柄は含まれない。

## テスト

```powershell
pytest tests/
```
