from __future__ import annotations

import time
from datetime import date

import pandas as pd
import requests

from .base import PriceDataSource

_BASE_URL = "https://api.jquants.com/v1"


class JQuantsAuth:
    """J-Quants のリフレッシュトークン→IDトークン交換を扱う。

    IDトークンは短命（24h程度）なため、有効期限が切れたら自動で再取得する。
    Phase 1 では未使用（フックのみ）。
    """

    def __init__(self, refresh_token: str):
        self._refresh_token = refresh_token
        self._id_token: str | None = None
        self._id_token_expires_at: float = 0.0

    def get_id_token(self) -> str:
        if self._id_token is None or time.time() >= self._id_token_expires_at:
            resp = requests.post(
                f"{_BASE_URL}/token/auth_refresh",
                params={"refreshtoken": self._refresh_token},
                timeout=30,
            )
            resp.raise_for_status()
            self._id_token = resp.json()["idToken"]
            self._id_token_expires_at = time.time() + 23 * 3600
        return self._id_token


class JQuantsSource(PriceDataSource):
    """J-Quants API アダプタのスタブ。

    Phase 2 で本実装する。インターフェースは `PriceDataSource` に準拠済みなので
    `config/settings.yaml` の `data_source: jquants` に切り替えるだけで
    呼び出し側コードの変更なしに利用できる設計にしてある。
    """

    name = "jquants"

    def __init__(self, auth: JQuantsAuth):
        self._auth = auth

    def fetch_ohlcv(
        self, ticker: str, start: date, end: date, interval: str = "1d"
    ) -> pd.DataFrame:
        raise NotImplementedError(
            "J-Quants 連携は Phase 2 で実装予定。現在は data_source: yfinance を使用してください。"
        )
