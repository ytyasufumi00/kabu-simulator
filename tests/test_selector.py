from __future__ import annotations

from stockrl.experiments.config import PromotionConfig
from stockrl.experiments.runner import FoldResult, VariantResult
from stockrl.experiments.selector import compare_to_champion


def _variant(name: str, sharpes: list[float]) -> VariantResult:
    return VariantResult(
        variant_name=name,
        fold_results=[
            FoldResult(fold_index=i, metrics={"sharpe_ratio": s}) for i, s in enumerate(sharpes)
        ],
    )


PROMOTION_CFG = PromotionConfig(min_fold_win_ratio=0.6, min_fold_margin=0.05)


def test_no_existing_champion_always_promotes() -> None:
    challenger = _variant("a", [0.1, 0.1, 0.1])
    decision = compare_to_champion(challenger, None, PROMOTION_CFG)
    assert decision.promote is True


def test_challenger_clearly_better_is_promoted() -> None:
    champion = _variant("champion", [0.1, 0.1, 0.1])
    challenger = _variant("challenger", [0.5, 0.5, 0.5])
    decision = compare_to_champion(challenger, champion, PROMOTION_CFG)
    assert decision.promote is True
    assert decision.win_ratio == 1.0


def test_challenger_worse_is_not_promoted() -> None:
    champion = _variant("champion", [0.5, 0.5, 0.5])
    challenger = _variant("challenger", [0.1, 0.1, 0.1])
    decision = compare_to_champion(challenger, champion, PROMOTION_CFG)
    assert decision.promote is False
    assert decision.win_ratio == 0.0


def test_challenger_with_insufficient_margin_is_not_promoted() -> None:
    champion = _variant("champion", [0.10, 0.10, 0.10])
    challenger = _variant("challenger", [0.12, 0.12, 0.12])  # margin=0.02 < 閾値0.05
    decision = compare_to_champion(challenger, champion, PROMOTION_CFG)
    assert decision.promote is False


def test_challenger_with_insufficient_win_ratio_is_not_promoted() -> None:
    # 1/3foldしか勝てない（win_ratio=0.33 < 閾値0.6）が、マージンだけ見ると平均は勝っている
    champion = _variant("champion", [0.10, 0.10, 0.10])
    challenger = _variant("challenger", [0.50, 0.05, 0.05])
    decision = compare_to_champion(challenger, champion, PROMOTION_CFG)
    assert decision.win_ratio < 0.6
    assert decision.promote is False


def test_mismatched_fold_count_raises() -> None:
    champion = _variant("champion", [0.1, 0.1])
    challenger = _variant("challenger", [0.1, 0.1, 0.1])
    try:
        compare_to_champion(challenger, champion, PROMOTION_CFG)
        assert False, "ValueErrorが発生するはず"
    except ValueError:
        pass
