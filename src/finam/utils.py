"""
Утилиты для работы с признаками и данными
"""

import pandas as pd


def get_feature_columns(df: pd.DataFrame, horizons: list[int] = None) -> list[str]:
    """
    Получить список колонок с features (исключая target, ticker, date и OHLCV)

    Args:
        df: DataFrame после add_all_features
        horizons: список горизонтов для исключения target_return_*d (по умолчанию 1-20)

    Returns:
        список названий колонок с features

    Example:
        >>> df = pd.DataFrame({
        ...     'ticker': ['SBER'],
        ...     'close': [100],
        ...     'momentum_5d': [0.02],
        ...     'target_return_1d': [0.01]
        ... })
        >>> get_feature_columns(df)
        ['momentum_5d']
    """
    if horizons is None:
        horizons = list(range(1, 21))

    # Базовые колонки которые НЕ являются features
    exclude_cols = {
        'ticker', 'begin', 'dataset',
        'open', 'high', 'low', 'close', 'volume'
    }

    # Добавляем все target_return_*d колонки
    for h in horizons:
        exclude_cols.add(f'target_return_{h}d')

    # Паттерны для исключения predictions
    exclude_patterns = ['pred_return_', 'pred_prob_up_']

    feature_cols = []
    for col in df.columns:
        # Проверяем точное совпадение
        if col in exclude_cols:
            continue
        # Проверяем паттерны
        if any(pattern in col for pattern in exclude_patterns):
            continue
        feature_cols.append(col)

    return feature_cols
