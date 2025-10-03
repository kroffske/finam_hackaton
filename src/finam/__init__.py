"""
finam package - модули для feature engineering и моделирования
"""

__version__ = "0.1.0"

# Expose key functions for easy import
from .features import add_all_features
from .metrics import directional_accuracy, mae, brier_score, normalized_score
from .model import BaseModel, MomentumBaseline, LightGBMModel, CalibratedLightGBMModel

__all__ = [
    "add_all_features",
    "mae",
    "brier_score",
    "directional_accuracy",
    "normalized_score",
    "BaseModel",
    "MomentumBaseline",
    "LightGBMModel",
    "CalibratedLightGBMModel",
]
