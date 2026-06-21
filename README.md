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
- PPOのロールアウト収集は`config/experiments.yaml`の`n_envs`（既定4）でSB3の`SubprocVecEnv`により
  並列化される。総timesteps（学習に使うデータ量）はn_envsに依存せず一定なので、学習の精度には
  影響せず壁時計時間のみが短縮される。

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

### Champion比較＆再学習トリガー（ダッシュボード上のボタン）

ダッシュボードには、各銘柄のChampion/Challenger比較履歴（`promotions.csv`）の一覧と、
クラウド学習（`train-cloud.yml`）をその場で起動できるボタンがある。

- このCloud Runサービスは`Allow unauthenticated access`設定で**URLを知っていれば誰でも閲覧できる**。
  起動ボタンは課金を伴う操作のため、ダッシュボード上の「管理者操作」欄でパスワード認証を要求する
  （簡易的な誤操作防止であり、強固なアクセス制御ではない）。
- 認証後にボタンを押すと、GitHub REST API（`POST /repos/.../actions/workflows/train-cloud.yml/dispatches`）
  経由でワークフローをdispatchする。GitHub Actions上で動くワークフロー自体は組み込みの`GITHUB_TOKEN`を
  使えるが、**Cloud Run上で動くStreamlitアプリから外部にdispatchするには別途PAT（Personal Access Token）
  が必要**なため、`Actions: Read and write`権限のfine-grained PATをSecret Manager経由で渡している。
- 必要なGCP Secret Manager上のシークレット（Cloud Runへは`deploy-cloudrun`の`secrets`入力で
  `DASHBOARD_ADMIN_SECRET` / `GH_DISPATCH_TOKEN`として注入）:
  - `dashboard-admin-secret`: ダッシュボードの管理者パスワード
  - `dashboard-gh-dispatch-token`: `kabu-simulator`リポジトリの`Actions: Read and write`権限を持つ
    fine-grained PAT。**有効期限が切れたら再作成し`gcloud secrets versions add dashboard-gh-dispatch-token
    --data-file=-`で更新が必要**。
  - 両シークレットはCloud Runランタイムサービスアカウント（`<project-number>-compute@developer.gserviceaccount.com`）
    と デプロイ用サービスアカウント（`kabu-simulator-deployer@...`）に`roles/secretmanager.secretAccessor`を付与済み。
- ダッシュボードには「前回の比較実行から何日経過したか」を表示し、14日以上経過していれば再実行を推奨する
  警告を出す（目安: 2週間〜1ヶ月、フォワードテストの結果がバックテストと大きくズレた時、月初のフォワード
  championスナップショット更新前）。

## クラウド学習（Spot VM）

`run_champion_loop_all_tickers`は数時間規模の処理になるため、PCを起動し続ける代わりに
GCPのSpot（プリエンプティブル）VMで実行できる。

```powershell
# GitHub ActionsのUIから手動実行（Actions タブ → "Run cloud training (Spot VM)" → Run workflow）
# または gh CLI から:
gh workflow run train-cloud.yml --repo ytyasufumi00/kabu-simulator

# 完了後、結果をローカルに同期
python -m stockrl.cli.sync_cloud_results
```

### 仕組みとコスト安全策

- 素のDebian VM（GCE公式イメージにgcloud/gsutilが標準搭載されているためコンテナ化不要）に
  startup-scriptを渡すだけ。スクリプトがclone・依存インストール・学習・GCSへの結果アップロード・
  自己削除まで一気通貫で行う（`training/startup-script.sh`）。
- コスト安全策を多重化: Spot価格（オンデマンドの60〜90%引き）、`--instance-termination-action=DELETE`
  （GCPにpreemptされても自動削除）、`--max-run-duration=21600s`（6時間で強制終了）、
  スクリプト内`trap`による正常/異常終了時の自己削除。
- 進行状況は`gcloud compute instances get-serial-port-output <インスタンス名> --zone=asia-northeast1-a`
  で確認できる。
- **VMの起動自体・GCS保存料金はGCPの計算課金（Claude APIとは別軸）が発生する**。Spot価格でも
  ゼロにはならないため、念のため定期的に`gcloud compute instances list`でVMが残っていないか確認すること。

## フォワードテスト（未来のデータでのchampion運用検証）

walk-forwardはあくまで過去データへの検証であり、過学習を否定できない。そこで、**championモデルを
1ヶ月単位で固定し、まだ結果のわからない未来の値動きに対して推論だけを行う**フォワードテスト
（ペーパートレード）を毎営業日自動実行する。

```powershell
# 手動実行（通常はGitHub Actionsのcronが平日自動実行する）
python -m stockrl.cli.run_forward_test
```

### 仕組み

- `config/forward_test.yaml`で対象銘柄・1銘柄あたりの仮想投資額を定義する
  （初期値: 100万円を5銘柄に20万円ずつ）。銘柄を追加する場合はここに追記する
  （既存銘柄からの引き出し・再配分はせず、新規の追加投資として扱う簡略化）。
- 月初の最初の実行で`runs/{ticker}/champion/`（バックテストのChampion/Challengerループが
  継続的に更新するもの）のスナップショットを`runs/{ticker}/forward_champion/`に複製する。
  以降1ヶ月間はこの固定モデルだけで推論を行う（バックテスト側のループ自体は今までどおり
  好きなタイミングで実行してよい。止まるのはフォワードテストの対象モデルだけ）。
- 観測ベクトルの生成は`env/single_asset_env.py`の`build_observation()`を直接呼び、
  学習時と完全に同じフォーマットを保証する。
- Portfolio状態（cash・保有株数）は`runs/{ticker}/forward_test/state.json`に永続化し、
  日次ログは`daily_log.csv`に追記する。状態はGCS（`kabu-simulator-runs-...`バケット、
  クラウド学習と共用）経由でGitHub Actionsの実行間を引き継ぐ。
- ダッシュボードでは全銘柄の評価額を合計し、「投資総額」と「評価額」の2本線で表示する
  （積立投資アプリでよく見る形式。銘柄追加のたびに投資総額が増える）。

### 実行タイミングとコスト

- `.github/workflows/forward-test-daily.yml`が平日JST16:00頃（東証引け後）に自動実行する。
- 学習ではなく推論のみのため、Spot VMは使わず通常のGitHub Actionsランナーで数分で完了する
  （public repoのため実行時間は無料）。GCSの保存料も日次の数KB程度でほぼ無視できる。
