"""Feature exclusion utilities used across preprocessing and training.

The default blacklist is derived from the LightGBM feature importance
report stored at
``outputs/2025-10-05_01-47-38_final_2/feature_importance.csv``.
All columns ranked below the top-50 by average importance were added to
``EXCLUDED_FEATURES`` so that they are dropped from future experiments by
default. Keeping the list in a dedicated module makes it easy to tweak or
extend during the hackathon without wiring extra configuration files.

Example:
    >>> cols = ["ma_5d", "llm_sentiment_mean", "volatility_20d"]
    >>> exclude_feature_columns(cols)
    ['ma_5d', 'volatility_20d']
"""

from __future__ import annotations

from typing import Iterable, Sequence


# Features to drop (sorted alphabetically for readability).
EXCLUDED_FEATURES: list[str] = sorted(
    [
        "distance_from_ma_20d_rank",
        "distance_from_ma_5d_rank",
        "llm_confidence_mean",
        "llm_confidence_mean_3d",
        "llm_confidence_sum",
        "llm_confidence_sum_14d",
        "llm_confidence_sum_30d",
        "llm_confidence_sum_3d",
        "llm_confidence_sum_60d",
        "llm_confidence_sum_7d",
        "llm_negative_count",
        "llm_negative_count_3d",
        "llm_negative_count_7d",
        "llm_news_count_14d",
        "llm_news_count_1d",
        "llm_news_count_30d",
        "llm_news_count_3d",
        "llm_news_count_60d",
        "llm_news_count_7d",
        "llm_news_type_count__company_specific",
        "llm_news_type_count__company_specific_14d",
        "llm_news_type_count__company_specific_30d",
        "llm_news_type_count__company_specific_3d",
        "llm_news_type_count__company_specific_60d",
        "llm_news_type_count__company_specific_7d",
        "llm_news_type_count__market_wide_company",
        "llm_news_type_count__market_wide_company_14d",
        "llm_news_type_count__market_wide_company_30d",
        "llm_news_type_count__market_wide_company_3d",
        "llm_news_type_count__market_wide_company_60d",
        "llm_news_type_count__market_wide_company_7d",
        "llm_news_type_share__company_specific",
        "llm_news_type_share__company_specific_14d",
        "llm_news_type_share__company_specific_30d",
        "llm_news_type_share__company_specific_3d",
        "llm_news_type_share__company_specific_60d",
        "llm_news_type_share__company_specific_7d",
        "llm_news_type_share__market_wide_company",
        "llm_news_type_share__market_wide_company_14d",
        "llm_news_type_share__market_wide_company_30d",
        "llm_news_type_share__market_wide_company_3d",
        "llm_news_type_share__market_wide_company_60d",
        "llm_news_type_share__market_wide_company_7d",
        "llm_neutral_count",
        "llm_neutral_count_3d",
        "llm_neutral_count_7d",
        "llm_positive_count",
        "llm_positive_count_3d",
        "llm_positive_count_7d",
        "llm_sentiment_mean",
        "llm_sentiment_mean_14d",
        "llm_sentiment_mean_3d",
        "llm_sentiment_weighted",
        "llm_sentiment_weighted_14d",
        "llm_sentiment_weighted_3d",
        "llm_sentiment_weighted_7d",
        "momentum_20d_rank",
        "momentum_5d_rank",
        "rsi_14d_rank",
        "volatility_5d_rank",
        "volume_ratio_20d_rank",
        "volume_ratio_5d_rank",
    ]
)


def get_excluded_features(extra: Sequence[str] | None = None) -> list[str]:
    """Return the default blacklist merged with optional extras."""

    if not extra:
        return EXCLUDED_FEATURES.copy()
    merged = set(EXCLUDED_FEATURES)
    merged.update(extra)
    return sorted(merged)


def exclude_feature_columns(
    feature_columns: Iterable[str],
    *,
    extra_exclude: Sequence[str] | None = None,
) -> list[str]:
    """Remove low-importance columns from ``feature_columns``.

    Args:
        feature_columns: Original iterable of feature names.
        extra_exclude: Additional columns to drop for a particular
            experiment.

    Returns:
        List with excluded names removed while keeping the original order of
        ``feature_columns``.
    """

    blacklist = set(get_excluded_features(extra_exclude))
    return [col for col in feature_columns if col not in blacklist]
