"""
Target Engineering для FORECAST задачи

Функции для вычисления таргетов доходности на разных горизонтах
"""

import numpy as np
import pandas as pd


def compute_multi_horizon_targets(
    df: pd.DataFrame,
    horizons: list[int] = None
) -> pd.DataFrame:
    """
    Вычисляет таргеты доходности для множественных горизонтов

    Args:
        df: DataFrame с колонками ['ticker', 'begin', 'close']
        horizons: список горизонтов (по умолчанию 1-20 дней)

    Returns:
        df с добавленными колонками target_return_{horizon}d

    Example:
        >>> df = pd.DataFrame({
        ...     'ticker': ['SBER', 'SBER', 'SBER'],
        ...     'close': [100, 102, 105]
        ... })
        >>> df = compute_multi_horizon_targets(df, horizons=[1, 2])
        >>> # target_return_1d[0] = (102 - 100) / 100 = 0.02
        >>> # target_return_2d[0] = (105 - 100) / 100 = 0.05
    """
    if horizons is None:
        horizons = list(range(1, 21))  # 1-20 дней по умолчанию

    df = df.copy()

    for horizon in horizons:
        col_name = f'target_return_{horizon}d'

        # Сдвигаем цену на horizon дней вперед
        # shift(-horizon) берет будущее значение
        df[col_name] = (
            df.groupby('ticker')['close'].shift(-horizon) / df['close'] - 1
        )

    return df


def get_target_columns(horizons: list[int] = None) -> list[str]:
    """
    Получить список названий колонок с таргетами

    Args:
        horizons: список горизонтов (по умолчанию 1-20)

    Returns:
        список названий колонок ['target_return_1d', 'target_return_2d', ...]
    """
    if horizons is None:
        horizons = list(range(1, 21))

    return [f'target_return_{h}d' for h in horizons]


def extract_targets_dict(df: pd.DataFrame, horizons: list[int] = None) -> dict:
    """
    Извлекает таргеты в формате dict для обучения модели

    Args:
        df: DataFrame с target_return_*d колонками
        horizons: список горизонтов (по умолчанию 1-20)

    Returns:
        dict {
            'target_return_1d': np.ndarray,
            'target_return_2d': np.ndarray,
            ...
        }
    """
    if horizons is None:
        horizons = list(range(1, 21))

    targets_dict = {}

    for horizon in horizons:
        col_name = f'target_return_{horizon}d'
        if col_name in df.columns:
            targets_dict[col_name] = df[col_name].values
        else:
            print(f"[WARN] Column {col_name} not found in DataFrame")

    return targets_dict
