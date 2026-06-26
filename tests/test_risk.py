from __future__ import annotations

from stockrl.env.risk import RiskLimits, apply_risk_overlay


def test_clamps_target_to_0_1_range() -> None:
    limits = RiskLimits(max_position_pct=1.0, target_volatility=1.0, stop_loss_drawdown_pct=1.0)
    assert apply_risk_overlay(1.5, volatility=0.0, peak_equity=0.0, current_equity=0.0, limits=limits) == 1.0
    assert apply_risk_overlay(-0.5, volatility=0.0, peak_equity=0.0, current_equity=0.0, limits=limits) == 0.0


def test_volatility_targeting_scales_down_in_volatile_periods() -> None:
    limits = RiskLimits(max_position_pct=1.0, target_volatility=0.02, stop_loss_drawdown_pct=1.0)
    # 目標ボラ2%に対して実際は4%なので、上限は半分の50%に抑えられる
    result = apply_risk_overlay(1.0, volatility=0.04, peak_equity=0.0, current_equity=0.0, limits=limits)
    assert result == 0.5


def test_volatility_targeting_does_not_increase_target() -> None:
    limits = RiskLimits(max_position_pct=1.0, target_volatility=0.02, stop_loss_drawdown_pct=1.0)
    # 値動きが穏やかでも、モデルの希望(30%)を超えて増やしはしない
    result = apply_risk_overlay(0.3, volatility=0.01, peak_equity=0.0, current_equity=0.0, limits=limits)
    assert result == 0.3


def test_max_position_pct_caps_allocation() -> None:
    limits = RiskLimits(max_position_pct=0.8, target_volatility=1.0, stop_loss_drawdown_pct=1.0)
    result = apply_risk_overlay(1.0, volatility=0.0, peak_equity=0.0, current_equity=0.0, limits=limits)
    assert result == 0.8


def test_stop_loss_forces_zero_allocation_on_large_drawdown() -> None:
    limits = RiskLimits(max_position_pct=1.0, target_volatility=1.0, stop_loss_drawdown_pct=0.15)
    # ピーク100万→現在80万で20%下落（15%基準を超過）
    result = apply_risk_overlay(
        1.0, volatility=0.0, peak_equity=1_000_000.0, current_equity=800_000.0, limits=limits
    )
    assert result == 0.0


def test_stop_loss_not_triggered_below_threshold() -> None:
    limits = RiskLimits(max_position_pct=1.0, target_volatility=1.0, stop_loss_drawdown_pct=0.15)
    # ピーク100万→現在90万で10%下落（15%基準未満）
    result = apply_risk_overlay(
        1.0, volatility=0.0, peak_equity=1_000_000.0, current_equity=900_000.0, limits=limits
    )
    assert result == 1.0
