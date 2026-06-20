from __future__ import annotations

from stockrl.env.portfolio import Portfolio


def test_buy_all_spends_all_cash_minus_commission() -> None:
    p = Portfolio(initial_cash=1000.0, commission_pct=0.01)
    p.buy_all(step=0, price=100.0)

    assert p.cash == 0.0
    assert p.shares_held > 0
    # 手数料込みで購入したので、評価額は初期資金よりわずかに少ない
    assert p.equity(price=100.0) < 1000.0
    assert p.equity(price=100.0) == p.shares_held * 100.0


def test_sell_all_converts_shares_to_cash() -> None:
    p = Portfolio(initial_cash=1000.0, commission_pct=0.0)
    p.buy_all(step=0, price=100.0)
    p.sell_all(step=1, price=110.0)

    assert p.shares_held == 0.0
    assert p.cash > 1000.0  # 値上がりで利益が出ている


def test_buy_with_no_cash_is_noop() -> None:
    p = Portfolio(initial_cash=0.0)
    p.buy_all(step=0, price=100.0)
    assert p.shares_held == 0.0
    assert len(p.trades) == 0


def test_record_equity_appends_history() -> None:
    p = Portfolio(initial_cash=1000.0)
    p.record_equity(100.0)
    p.record_equity(105.0)
    assert p.equity_history == [1000.0, 1000.0]
