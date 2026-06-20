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
- `run_growth_loop`系（単一モデルを伸ばす用途）は固定の1分割のみ（walk-forwardは未実装）。
  単一期間での結果は「パイプラインが動くか」の確認であり、「戦略が優れているか」の結論には
  使えない。walk-forwardでの比較は次節の`run_champion_loop`で行う。
- 1モデル=1銘柄。複数銘柄を1つのRL環境で同時に扱うマルチアセット環境は未実装
  （`src/stockrl/env/multi_asset_env.py` にスタブのみ）。
- データソースはyfinanceのみ実装。J-Quantsはインターフェース・認証スタブのみ
  （`src/stockrl/data/jquants_source.py`）。
- サバイバーシップバイアス: 対象銘柄は現在上場している大型株であり、過去に上場廃止した
  銘柄は含まれない。

## 自動評価・自動更新（Champion/Challenger）

「どの強化（reward関数・ハイパラ等）が優れているか」をwalk-forwardで自動的に比較し、
優れたものを自動的に採用（昇格）する仕組み。**判定ロジックは完全に決定的なPython統計比較
であり、Claude等のLLM APIは一切呼び出さない**（課金・APIキー不要）。

```powershell
# 1銘柄でvariant（config/experiments.yamlで定義）をwalk-forward評価し、championを自動更新
python -m stockrl.cli.run_champion_loop --ticker 7203.T

# settings.yamlの全銘柄について実行
python -m stockrl.cli.run_champion_loop_all_tickers
```

### 仕組み

- `config/experiments.yaml`にreward関数・PPOハイパラの候補（variant）を列挙
- 各variantを`splits.walk_forward_splits()`の全foldで**ゼロから**学習・評価
  （fold間の継続学習はしない。variant同士を公平に比較するため）
- 各fold内ではSB3の`EvalCallback`でベストチェックポイントを保持（早期終了に相当）
- 全variant中もっとも平均シャープレシオが高いものを「ベストchallenger」として選出
- `runs/{ticker}/champion/`にある既存championと比較し、**過半数以上のfoldで勝ち、かつ
  平均シャープレシオの差が閾値（`min_fold_margin`）以上**の場合のみ自動昇格
- 昇格判定の結果（勝敗にかかわらず）は`runs/{ticker}/promotions.csv`に記録
- `snapshot_dashboard`は`champion/`が存在する銘柄ではそれを優先してダッシュボードに反映する

### 既知の簡略化

- `EvalCallback`のeval_envはそのfoldのtest splitそのものを使うため、厳密には学習中にtestを
  覗っている（チェックポイント選択バイアス）。train/val/testの3分割への拡張は今後の課題。
- variantは手動定義された候補リストであり、Optuna等によるベイズ最適化のような自動探索は
  未実装（次フェーズ）。
- 連続的なポジションサイジングや複数銘柄間の知識共有も未対応（行動空間・環境設計自体を
  変える必要があるため、今回は対象外）。

## テスト

```powershell
pytest tests/
```

## 結果ダッシュボード（Cloud Run）

`runs/`の結果を軽量データとして切り出し、Streamlitダッシュボードとして表示する。

```powershell
# runs/ から最新結果を dashboard/data/ にスナップショットとして切り出す
python -m stockrl.cli.snapshot_dashboard

# ローカルで確認
cd dashboard
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

`dashboard/**`をmasterにpushすると、GitHub Actions（`.github/workflows/deploy-dashboard.yml`）が
Workload Identity Federation経由でGCPプロジェクト`project-1efc0d32-8b57-40c7-b3a`（表示名: ckd-prediction-app）
のCloud Run（リージョン: asia-northeast1, サービス名: kabu-simulator-dashboard）に自動デプロイする。
表示データは学習のたびに自動更新されるのではなく、`snapshot_dashboard`を再実行してcommit&pushした時点の
スナップショット。
