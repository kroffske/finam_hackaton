"""
finam package - feature engineering utilities for the FORECAST task.
"""

__version__ = "0.1.0"

# Expose key functions for easy import
from .features import add_all_features
from .features_target import compute_multi_horizon_targets, extract_targets_dict, get_target_columns
from .utils import get_feature_columns
from .metrics import mae
from .model import BaseModel, MomentumBaseline, LightGBMModel
from .news_tickers import (
    DEFAULT_TICKER_NAMES,
    assign_news_tickers,
    explode_news_tickers,
    find_tickers_in_text,
    normalize_text,
)

__all__ = [
    "add_all_features",
    "compute_multi_horizon_targets",
    "extract_targets_dict",
    "get_target_columns",
    "get_feature_columns",
    "mae",
    "BaseModel",
    "MomentumBaseline",
    "LightGBMModel",
    "DEFAULT_TICKER_NAMES",
    "find_tickers_in_text",
    "assign_news_tickers",
    "explode_news_tickers",
    "normalize_text",
]
