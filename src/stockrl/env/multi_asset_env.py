"""複数銘柄を1つのRL環境で同時に扱うマルチアセット環境（Phase 2、未実装）。

単一銘柄環境（`single_asset_env.SingleAssetTradingEnv`）と異なり、以下が
追加で必要になり設計上の難易度が上がるため、Phase 1ではあえて分離している:

- 複数銘柄間で共有される現金プール（同時に複数銘柄を買う際の資金配分問題）
- 行動空間がベクトル化される（各銘柄ごとにHold/Buy/Sellを同時に決定）
- 観測空間が銘柄数に比例して大きくなる（Dict observation か stacked Box）
- 銘柄間の相関を考慮したリスク管理（一つの銘柄の暴落が全体に与える影響）

Phase 1 では `SingleAssetTradingEnv` を銘柄ごとに独立して学習・評価し、
このアプローチがそもそも機能するかを先に検証する。
"""

from __future__ import annotations

import gymnasium as gym


class MultiAssetTradingEnv(gym.Env):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("マルチアセット環境はPhase 2で実装予定")
