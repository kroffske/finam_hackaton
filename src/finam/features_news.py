"""
News Features для FORECAST задачи

ВАЖНО: Новости доступны до t-1 (задержка 1 день), поэтому джойним с лагом!

Фичи:
- news_count_1d_lag: количество новостей за предыдущий день
- news_count_7d_lag: количество новостей за последние 7 дней (с лагом)
"""

import pandas as pd
import numpy as np


def compute_daily_news_count(
    news_df: pd.DataFrame,
    date_col: str = 'publish_date'
) -> pd.DataFrame:
    """
    Агрегация новостей по дням

    Args:
        news_df: DataFrame с новостями (должен содержать колонку date_col)
        date_col: название колонки с датой публикации

    Returns:
        DataFrame с колонками ['date', 'news_count']

    Example:
        date       news_count
        2020-01-01    15
        2020-01-02    23
        ...
    """
    # Копия для безопасности
    news = news_df.copy()

    # Преобразуем дату в datetime
    news[date_col] = pd.to_datetime(news[date_col])

    # Извлекаем дату (без времени)
    news['date'] = news[date_col].dt.date

    # Группируем по дате и считаем количество новостей
    daily_counts = news.groupby('date').size().reset_index(name='news_count')

    # Преобразуем date обратно в datetime для джойна
    daily_counts['date'] = pd.to_datetime(daily_counts['date'])

    return daily_counts


def add_news_features(
    candles_df: pd.DataFrame,
    news_df: pd.DataFrame,
    lag_days: int = 1,
    rolling_windows: list[int] = None
) -> pd.DataFrame:
    """
    Добавить новостные фичи к DataFrame со свечами

    ВАЖНО: Данные джойнятся с лагом lag_days для избежания data leakage!

    Логика:
    - Новости доступны до t-1
    - Для свечей за день t используем новости до t-1 включительно
    - Поэтому джойним candles['begin'] с news['date'] - lag_days

    Args:
        candles_df: DataFrame со свечами (колонка 'begin' = дата)
        news_df: DataFrame с новостями (колонка 'publish_date')
        lag_days: сдвиг новостей назад (по умолчанию 1 день)
        rolling_windows: список окон для rolling count (по умолчанию [1, 7, 30])

    Returns:
        candles_df с добавленными колонками:
        - news_count_1d_lag
        - news_count_7d_lag
        - news_count_30d_lag

    Example:
        >>> candles = pd.read_csv('train_candles.csv')
        >>> news = pd.read_csv('train_news.csv')
        >>> candles = add_news_features(candles, news, lag_days=1)
        >>> # Теперь candles содержит новостные фичи с безопасным лагом
    """
    if rolling_windows is None:
        rolling_windows = [1, 7, 30]

    print("   * News features...")

    # 1. Агрегация новостей по дням
    daily_news = compute_daily_news_count(news_df, date_col='publish_date')

    # 2. Подготовка candles
    candles = candles_df.copy()
    candles['begin'] = pd.to_datetime(candles['begin'])

    # 3. Создаём полный date range для безопасного join
    # (чтобы не потерять дни без новостей)
    min_date = candles['begin'].min()
    max_date = candles['begin'].max()

    # Расширяем диапазон для учёта лага и rolling windows
    date_range_start = min_date - pd.Timedelta(days=max(rolling_windows) + lag_days)
    date_range_end = max_date

    all_dates = pd.DataFrame({
        'date': pd.date_range(start=date_range_start, end=date_range_end, freq='D')
    })

    # Merge с новостями (left join чтобы сохранить все даты)
    all_dates = all_dates.merge(daily_news, on='date', how='left')
    all_dates['news_count'] = all_dates['news_count'].fillna(0).astype(int)

    # 4. Вычисляем rolling counts
    for window in rolling_windows:
        col_name = f'news_count_{window}d_rolling'
        all_dates[col_name] = all_dates['news_count'].rolling(
            window=window,
            min_periods=1
        ).sum().astype(int)

    # 5. Применяем LAG (ключевой момент!)
    # Для свечей за день t используем новости до t-lag_days
    all_dates['date_shifted'] = all_dates['date'] + pd.Timedelta(days=lag_days)

    # 6. Join к candles
    # candles['begin'] == all_dates['date_shifted']
    # Это означает: для свечей за день t берём новости за день t-lag_days
    merge_cols = ['date_shifted'] + [f'news_count_{w}d_rolling' for w in rolling_windows]

    candles = candles.merge(
        all_dates[merge_cols],
        left_on='begin',
        right_on='date_shifted',
        how='left'
    )

    # Cleanup temporary column
    candles = candles.drop(columns=['date_shifted'])

    # 7. Переименовываем колонки с лагом
    for window in rolling_windows:
        old_name = f'news_count_{window}d_rolling'
        new_name = f'news_count_{window}d_lag'
        candles = candles.rename(columns={old_name: new_name})

    # 8. Заполняем NaN нулями (дни до начала новостей)
    for window in rolling_windows:
        col_name = f'news_count_{window}d_lag'
        if col_name in candles.columns:
            candles[col_name] = candles[col_name].fillna(0).astype(int)

    print(f"      OK Added {len(rolling_windows)} news features with {lag_days}-day lag")

    return candles


def get_news_feature_columns(rolling_windows: list[int] = None) -> list[str]:
    """
    Получить список названий новостных фич

    Args:
        rolling_windows: список окон (по умолчанию [1, 7, 30])

    Returns:
        список названий колонок

    Example:
        >>> get_news_feature_columns()
        ['news_count_1d_lag', 'news_count_7d_lag', 'news_count_30d_lag']
    """
    if rolling_windows is None:
        rolling_windows = [1, 7, 30]

    return [f'news_count_{w}d_lag' for w in rolling_windows]
