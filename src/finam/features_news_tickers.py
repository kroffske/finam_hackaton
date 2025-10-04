"""
Ticker-level News Features для FORECAST задачи

Создает признаки по новостям для каждого тикера на уровне (date, ticker).

ВАЖНО:
- Входные данные уже exploded (одна строка = один тикер)
- Новости доступны до t-1 (задержка 1 день)
- Признаки создаются на уровне (date, ticker)

Пример использования:
    >>> ticker_sentiment = pd.read_csv('news_ticker_sentiment.csv')  # exploded
    >>> ticker_features = aggregate_ticker_news_features(ticker_sentiment)
    >>> # Результат: (date, ticker, llm_news_count_1d, llm_sentiment_mean, ...)
"""

import re
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


def _sanitize_category_value(value: object) -> str:
    """Convert category value to safe snake_case token."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 'nan'

    sanitized = re.sub(r'[^0-9a-zA-Z]+', '_', str(value).strip())
    sanitized = sanitized.strip('_').lower()
    return sanitized or 'unknown'


def _rolling_feature_name(column: str, window: int) -> str:
    """Create consistent rolling window feature name."""
    if column.endswith('_1d'):
        base = column[:-3]
    else:
        base = column
    return f"{base}_{window}d"


def compute_ticker_daily_counts(ticker_sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Подсчитать количество новостей по (date, ticker)

    Args:
        ticker_sentiment_df: DataFrame с колонками [publish_date, ticker]

    Returns:
        DataFrame с колонками [date, ticker, llm_news_count_1d]
    """
    df = ticker_sentiment_df.copy()

    # Преобразовать дату
    df['publish_date'] = pd.to_datetime(df['publish_date'])
    df['date'] = df['publish_date'].dt.date
    df['date'] = pd.to_datetime(df['date'])

    # Группировка по (date, ticker)
    daily_counts = df.groupby(['date', 'ticker']).size().reset_index(name='llm_news_count_1d')

    return daily_counts


def add_rolling_features(
    df: pd.DataFrame,
    windows: Iterable[int],
    sum_columns: Iterable[str],
    mean_columns: Iterable[str]
) -> pd.DataFrame:
    """Добавить rolling статистики для выбранных колонок."""
    result = df.sort_values(['ticker', 'date']).copy()

    windows = list(windows)

    for window in windows:
        if window <= 0:
            continue

        if sum_columns:
            for column in sum_columns:
                if column not in result.columns:
                    continue

                new_col = _rolling_feature_name(column, window)
                result[new_col] = (
                    result.groupby('ticker')[column]
                    .transform(
                        lambda values: values.astype(float).rolling(
                            window=window,
                            min_periods=1
                        ).sum()
                    )
                )

        if mean_columns:
            for column in mean_columns:
                if column not in result.columns:
                    continue

                new_col = _rolling_feature_name(column, window)
                result[new_col] = (
                    result.groupby('ticker')[column]
                    .transform(
                        lambda values: values.astype(float).rolling(
                            window=window,
                            min_periods=1
                        ).mean()
                    )
                )

    return result


def aggregate_categorical_features(
    ticker_sentiment_df: pd.DataFrame,
    column: str
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Сгенерировать счётчики и доли для категориального признака."""
    if column not in ticker_sentiment_df.columns:
        return pd.DataFrame(columns=['date', 'ticker']), {}

    df = ticker_sentiment_df[['publish_date', 'ticker', column]].copy()
    df['publish_date'] = pd.to_datetime(df['publish_date'])
    df['date'] = df['publish_date'].dt.normalize()

    df[column] = df[column].fillna('missing')

    # Bool -> str, числа -> str для унификации
    if is_bool_dtype(df[column]) or is_numeric_dtype(df[column]):
        df[column] = df[column].astype(str)

    counts = (
        df.groupby(['date', 'ticker', column])
        .size()
        .unstack(fill_value=0)
    )

    if counts.empty:
        return pd.DataFrame(columns=['date', 'ticker']), {}

    count_columns = [
        f"llm_{column}_count__{_sanitize_category_value(cat)}"
        for cat in counts.columns
    ]
    counts.columns = count_columns

    shares = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    share_columns = [col.replace('_count__', '_share__') for col in count_columns]
    shares.columns = share_columns

    combined = pd.concat([counts, shares], axis=1).reset_index()

    feature_roles = {col: 'sum' for col in count_columns}
    feature_roles.update({col: 'mean' for col in share_columns})

    return combined, feature_roles


def add_llm_sentiment_features(
    ticker_sentiment_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Агрегация sentiment/ confidence признаков на уровне (date, ticker)."""
    if 'sentiment' not in ticker_sentiment_df.columns:
        raise ValueError("DataFrame must contain 'sentiment' column")

    df = ticker_sentiment_df.copy()

    df['publish_date'] = pd.to_datetime(df['publish_date'])
    df['date'] = df['publish_date'].dt.normalize()

    df['sentiment'] = pd.to_numeric(df['sentiment'], errors='coerce')

    if 'confidence' in df.columns:
        df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce')
    else:
        df['confidence'] = np.nan

    df['__positive'] = (df['sentiment'] > 0).astype(int)
    df['__negative'] = (df['sentiment'] < 0).astype(int)
    df['__neutral'] = (df['sentiment'] == 0).astype(int)
    df['__sentiment_conf_product'] = df['sentiment'] * df['confidence']

    grouped = df.groupby(['date', 'ticker'])

    features = grouped['sentiment'].mean().rename('llm_sentiment_mean').to_frame()
    feature_roles: Dict[str, str] = {'llm_sentiment_mean': 'mean'}

    features['llm_positive_count'] = grouped['__positive'].sum()
    features['llm_negative_count'] = grouped['__negative'].sum()
    features['llm_neutral_count'] = grouped['__neutral'].sum()
    feature_roles.update({
        'llm_positive_count': 'sum',
        'llm_negative_count': 'sum',
        'llm_neutral_count': 'sum',
    })

    features['llm_confidence_mean'] = grouped['confidence'].mean()
    features['llm_confidence_sum'] = grouped['confidence'].sum()
    feature_roles.update({
        'llm_confidence_mean': 'mean',
        'llm_confidence_sum': 'sum',
    })

    numerator = grouped['__sentiment_conf_product'].sum()
    denominator = features['llm_confidence_sum'].replace(0, np.nan)
    weighted = numerator / denominator
    features['llm_sentiment_weighted'] = weighted.fillna(features['llm_sentiment_mean']).fillna(0.0)
    feature_roles['llm_sentiment_weighted'] = 'mean'

    # Cleanup helper columns
    features = features.reset_index()

    count_cols = ['llm_positive_count', 'llm_negative_count', 'llm_neutral_count']
    for column in count_cols:
        features[column] = features[column].fillna(0).astype(int)

    mean_cols = ['llm_sentiment_mean', 'llm_sentiment_weighted', 'llm_confidence_mean']
    for column in mean_cols:
        features[column] = features[column].fillna(0.0)

    features['llm_confidence_sum'] = features['llm_confidence_sum'].fillna(0.0)

    return features, feature_roles


def aggregate_ticker_news_features(
    ticker_sentiment_df: pd.DataFrame,
    rolling_windows: list[int] = None,
    include_llm_features: bool = True
) -> pd.DataFrame:
    """
    Агрегировать ticker-level признаки из exploded ticker-sentiment данных.

    Args:
        ticker_sentiment_df: DataFrame с колонками [publish_date, ticker, sentiment?, confidence?]
                             (уже exploded - одна строка на тикер)
        rolling_windows: окна для rolling features (по умолчанию [3, 7, 14, 30, 60])
        include_llm_features: добавить признаки из sentiment/LLM колонок

    Returns:
        DataFrame с признаками на уровне (date, ticker) с префиксом `llm_`
    """
    if rolling_windows is None:
        rolling_windows = [3, 7, 14, 30, 60]

    rolling_windows = sorted({window for window in rolling_windows if window > 0})
    if not rolling_windows:
        rolling_windows = [3, 7, 14, 30, 60]

    # Подсчёт новостей по дням (базовый count)
    features = compute_ticker_daily_counts(ticker_sentiment_df)
    feature_roles: Dict[str, str] = {'llm_news_count_1d': 'sum'}

    df = ticker_sentiment_df.copy()
    if not df.empty:
        df['publish_date'] = pd.to_datetime(df['publish_date'])
        df['date'] = df['publish_date'].dt.normalize()

    if include_llm_features and 'sentiment' in df.columns:
        llm_features, llm_roles = add_llm_sentiment_features(df)
        features = features.merge(llm_features, on=['date', 'ticker'], how='left')
        feature_roles.update(llm_roles)

    # primary_ticker как числовой индикатор (если возможно)
    categorical_candidates: List[str] = []
    if 'primary_ticker' in df.columns:
        primary_numeric = pd.to_numeric(df['primary_ticker'], errors='coerce')
        if primary_numeric.notna().any():
            df['__primary_numeric'] = primary_numeric
            grouped_primary = df.groupby(['date', 'ticker'])['__primary_numeric']
            primary_stats = grouped_primary.agg(['sum', 'mean']).rename(
                columns={'sum': 'llm_primary_ticker_count', 'mean': 'llm_primary_ticker_share'}
            ).reset_index()

            features = features.merge(primary_stats, on=['date', 'ticker'], how='left')
            feature_roles['llm_primary_ticker_count'] = 'sum'
            feature_roles['llm_primary_ticker_share'] = 'mean'
        else:
            categorical_candidates.append('primary_ticker')

    # Прочие категориальные признаки
    for col in ['news_type', 'impact_scope']:
        if col in df.columns:
            categorical_candidates.append(col)

    for cat_column in categorical_candidates:
        cat_features, cat_roles = aggregate_categorical_features(df, cat_column)
        if not cat_features.empty:
            features = features.merge(cat_features, on=['date', 'ticker'], how='left')
            feature_roles.update(cat_roles)

    # Заполнение пропусков
    for column, role in feature_roles.items():
        if column not in features.columns:
            continue
        if role == 'sum':
            features[column] = features[column].fillna(0.0)
        else:
            features[column] = features[column].fillna(0.0)

    sum_columns = [col for col, role in feature_roles.items() if role == 'sum']
    mean_columns = [col for col, role in feature_roles.items() if role == 'mean']

    if rolling_windows:
        features = add_rolling_features(
            features,
            windows=rolling_windows,
            sum_columns=sum_columns,
            mean_columns=mean_columns,
        )

    count_base_columns = [
        column for column in sum_columns
        if column.endswith('_count') or '_count__' in column or column == 'llm_news_count_1d'
    ]

    for column in count_base_columns:
        if column in features.columns:
            features[column] = features[column].round().astype(int)
        for window in rolling_windows:
            rolled_column = _rolling_feature_name(column, window)
            if rolled_column in features.columns:
                features[rolled_column] = features[rolled_column].round().astype(int)

    features = features.sort_values(['ticker', 'date']).reset_index(drop=True)

    return features


def join_ticker_news_features(
    candles_df: pd.DataFrame,
    ticker_news_features_df: pd.DataFrame,
    lag_days: int = 1
) -> pd.DataFrame:
    """
    Присоединить ticker news features к свечам с лагом

    ВАЖНО: Новости доступны до t-1, поэтому джойним с лагом!

    Args:
        candles_df: DataFrame со свечами [ticker, begin, ...]
        ticker_news_features_df: DataFrame с признаками [date, ticker, news_count_*]
        lag_days: лаг для избежания data leakage (по умолчанию 1)

    Returns:
        candles_df с добавленными новостными признаками

    Example:
        >>> candles = pd.read_csv('train.csv')
        >>> news_features = pd.read_csv('news_ticker_features.csv')
        >>> candles = join_ticker_news_features(candles, news_features, lag_days=1)
    """
    candles = candles_df.copy()
    news = ticker_news_features_df.copy()

    # Подготовка дат
    candles['begin'] = pd.to_datetime(candles['begin'])
    news['date'] = pd.to_datetime(news['date'])

    # Применяем лаг: для свечей за день t используем новости до t-lag_days
    news['date_shifted'] = news['date'] + pd.Timedelta(days=lag_days)

    # Join по (ticker, date)
    # candles['ticker', 'begin'] + news['ticker', 'date_shifted']
    feature_columns = [
        col for col in news.columns
        if col.startswith('llm_')
        or col.startswith('news_count_')  # fallback для старых файлов
        or col.startswith('sentiment_')
        or col.startswith('confidence_')
        or col.startswith('positive_count')
        or col.startswith('negative_count')
        or col.startswith('neutral_count')
        or col.startswith('topic_')
    ]

    news_cols = ['ticker', 'date_shifted'] + feature_columns

    candles = candles.merge(
        news[news_cols],
        left_on=['ticker', 'begin'],
        right_on=['ticker', 'date_shifted'],
        how='left'
    )

    # Cleanup
    candles = candles.drop(columns=['date_shifted'])

    # Заполнить NaN нулями для count колонок (дни без новостей)
    count_cols = [
        col for col in candles.columns
        if (
            col.startswith('llm_') and 'count' in col
        )
        or col.startswith('news_count_')
        or col.startswith('positive_count')
        or col.startswith('negative_count')
        or col.startswith('neutral_count')
    ]
    for col in count_cols:
        candles[col] = candles[col].fillna(0).astype(int)

    float_fill_cols = [
        col for col in candles.columns
        if (
            col.startswith('llm_')
            and (
                '_share' in col
                or '_mean' in col
                or '_weighted' in col
                or '_sum' in col
            )
        )
        or col.startswith('sentiment_')
        or col.startswith('confidence_')
    ]
    for col in float_fill_cols:
        candles[col] = candles[col].fillna(0.0)

    return candles
