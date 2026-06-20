from __future__ import annotations

from dataclasses import dataclass

from .config import PromotionConfig
from .runner import VariantResult


@dataclass(frozen=True)
class PromotionDecision:
    promote: bool
    win_ratio: float
    margin: float
    reason: str


def compare_to_champion(
    challenger: VariantResult,
    champion: VariantResult | None,
    promotion_cfg: PromotionConfig,
) -> PromotionDecision:
    """challengerをchampionと比較し、昇格させるべきかを決定的に判定する。

    APIコール・LLM判断は一切使わない。fold単位でのシャープレシオ比較に基づく
    純粋な数値ロジック。championが存在しない場合は無条件でchallengerを採用する。
    """
    if champion is None:
        return PromotionDecision(
            promote=True, win_ratio=1.0, margin=float("inf"), reason="既存championが存在しない"
        )

    if len(challenger.fold_results) != len(champion.fold_results):
        raise ValueError(
            "challengerとchampionのfold数が一致しない。同じwalk-forward設定で評価すること。"
        )

    n_folds = len(challenger.fold_results)
    if n_folds == 0:
        return PromotionDecision(promote=False, win_ratio=0.0, margin=0.0, reason="foldが0件")

    wins = sum(
        1
        for c, ch in zip(challenger.fold_results, champion.fold_results)
        if c.metrics["sharpe_ratio"] > ch.metrics["sharpe_ratio"]
    )
    win_ratio = wins / n_folds
    margin = challenger.mean_sharpe - champion.mean_sharpe

    promote = win_ratio >= promotion_cfg.min_fold_win_ratio and margin >= promotion_cfg.min_fold_margin

    reason = (
        f"win_ratio={win_ratio:.2f} (閾値{promotion_cfg.min_fold_win_ratio:.2f}), "
        f"margin={margin:+.4f} (閾値{promotion_cfg.min_fold_margin:.4f})"
    )
    return PromotionDecision(promote=promote, win_ratio=win_ratio, margin=margin, reason=reason)
