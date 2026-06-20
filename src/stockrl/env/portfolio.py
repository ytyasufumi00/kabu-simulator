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
    """単一銘柄の仮想ポートフォリオ（全部買い/全部売りの単純モデル）。

    手数料は `commission_pct` を取引のたびにノーション額に対して課す。
    スリッページは `slippage_bps` フックを用意しているが v1 は 0。
    """

    initial_cash: float
    commission_pct: float = 0.0
    slippage_bps: float = 0.0

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

    def buy_all(self, step: int, price: float) -> None:
        if self.cash <= 0:
            return
        exec_price = self._execution_price(price, is_buy=True)
        shares = self.cash / (exec_price * (1 + self.commission_pct))
        cost = shares * exec_price * self.commission_pct
        self.shares_held += shares
        self.cash = 0.0
        self.trades.append(Trade(step, "buy", exec_price, shares, cost))

    def sell_all(self, step: int, price: float) -> None:
        if self.shares_held <= 0:
            return
        exec_price = self._execution_price(price, is_buy=False)
        proceeds = self.shares_held * exec_price
        cost = proceeds * self.commission_pct
        self.cash += proceeds - cost
        self.trades.append(Trade(step, "sell", exec_price, self.shares_held, cost))
        self.shares_held = 0.0

    def record_equity(self, price: float) -> float:
        equity = self.equity(price)
        self.equity_history.append(equity)
        return equity
