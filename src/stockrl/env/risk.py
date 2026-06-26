from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskLimits:
    """モデルの判断に被せる安全装置のパラメータ。"""

    max_position_pct: float = 0.9
    target_volatility: float = 0.02
    stop_loss_drawdown_pct: float = 0.15


def apply_risk_overlay(
    target_pct: float,
    volatility: float,
    peak_equity: float,
    current_equity: float,
    limits: RiskLimits,
) -> float:
    """モデルが出した目標配分（target_pct）に安全装置を適用して上書きする。

    学習・フォワードテストの両方で同じロジックを通すことで、
    モデルが「実際に運用される制約込みの世界」で学習・評価される。

    1. ボラティリティターゲティング: 値動きが荒い時期は配分の上限を下げる
    2. 最大保有比率: 常に一定のキャッシュバッファを残す
    3. ストップロス: 直近ピークからの下落が大きい場合は強制的に配分をゼロにする
    """
    target_pct = max(0.0, min(1.0, target_pct))

    if volatility > 0:
        vol_scale = min(1.0, limits.target_volatility / volatility)
        target_pct = min(target_pct, vol_scale)

    target_pct = min(target_pct, limits.max_position_pct)

    if peak_equity > 0:
        drawdown = 1.0 - (current_equity / peak_equity)
        if drawdown >= limits.stop_loss_drawdown_pct:
            target_pct = 0.0

    return target_pct
