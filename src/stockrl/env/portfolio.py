from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Trade:
    step: int
    action: str  # "buy" | "sell"
    price: float
    shares: float
    cost: float


@dataclass
class Portfolio:
    """単一銘柄の仮想ポートフォリオ（目標配分への部分リバランス方式）。

    モデルは毎ステップ「保有比率を何%にすべきか（target_pct）」を出力し、
    現在の保有比率との差分だけを売買する（全部買い/全部売りの単純モデルではない）。
    手数料は `commission_pct` を取引のたびにノーション額に対して課す。
    スリッページは `slippage_bps` フックを用意しているが v1 は 0。
    """

    initial_cash: float
    commission_pct: float = 0.0
    slippage_bps: float = 0.0
    min_trade_pct: float = 0.0  # 目標配分との差がこの割合未満なら取引しない（手数料の無駄打ち防止）

    cash: float = field(init=False)
    shares_held: float = field(init=False, default=0.0)
    equity_history: list[float] = field(init=False, default_factory=list)
    trades: list[Trade] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.cash = self.initial_cash

    def _execution_price(self, price: float, is_buy: bool) -> float:
        slip = price * (self.slippage_bps / 10_000.0)
        return price + slip if is_buy else price - slip

    def equity(self, price: float) -> float:
        return self.cash + self.shares_held * price

    def position_pct(self, price: float) -> float:
        equity = self.equity(price)
        if equity <= 0:
            return 0.0
        return (self.shares_held * price) / equity

    def rebalance_to(self, step: int, target_pct: float, price: float) -> None:
        """目標配分（0.0〜1.0）に向けて保有株を部分的に売買する。

        現在の配分との差が `min_trade_pct` 未満なら、手数料の無駄打ちを避けるため何もしない。
        """
        equity = self.equity(price)
        if equity <= 0:
            return

        target_pct = max(0.0, min(1.0, target_pct))
        current_value = self.shares_held * price
        target_value = equity * target_pct
        diff_value = target_value - current_value

        if abs(diff_value) < equity * self.min_trade_pct:
            return

        if diff_value > 0:
            self._buy_value(step, price, min(diff_value, self.cash))
        else:
            self._sell_value(step, price, min(-diff_value, current_value))

    def _buy_value(self, step: int, price: float, value: float) -> None:
        if value <= 0 or self.cash <= 0:
            return
        exec_price = self._execution_price(price, is_buy=True)
        shares = value / (exec_price * (1 + self.commission_pct))
        cost = shares * exec_price * self.commission_pct
        self.shares_held += shares
        self.cash -= value
        self.trades.append(Trade(step, "buy", exec_price, shares, cost))

    def _sell_value(self, step: int, price: float, value: float) -> None:
        if value <= 0 or self.shares_held <= 0:
            return
        exec_price = self._execution_price(price, is_buy=False)
        shares = min(value / price, self.shares_held)
        proceeds = shares * exec_price
        cost = proceeds * self.commission_pct
        self.cash += proceeds - cost
        self.shares_held -= shares
        self.trades.append(Trade(step, "sell", exec_price, shares, cost))

    def record_equity(self, price: float) -> float:
        equity = self.equity(price)
        self.equity_history.append(equity)
        return equity
