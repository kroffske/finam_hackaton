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
    >>> # Результат: (date, ticker, news_count_1d, sentiment_mean, etc.)
"""

import pandas as pd


def compute_ticker_daily_counts(ticker_sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """
    Подсчитать количество новостей по (date, ticker)

    Args:
        ticker_sentiment_df: DataFrame с колонками [publish_date, ticker]

    Returns:
        DataFrame с колонками [date, ticker, news_count_1d]
    """
    df = ticker_sentiment_df.copy()

    # Преобразовать дату
    df['publish_date'] = pd.to_datetime(df['publish_date'])
    df['date'] = df['publish_date'].dt.date
    df['date'] = pd.to_datetime(df['date'])

    # Группировка по (date, ticker)
    daily_counts = df.groupby(['date', 'ticker']).size().reset_index(name='news_count_1d')

    return daily_counts


def add_rolling_features(
    daily_counts: pd.DataFrame,
    windows: list[int] = None
) -> pd.DataFrame:
    """
    Добавить rolling counts для каждого тикера

    Args:
        daily_counts: DataFrame с [date, ticker, news_count_1d]
        windows: список окон для rolling (по умолчанию [7, 30])

    Returns:
        DataFrame с добавленными колонками news_count_7d, news_count_30d
    """
    if windows is None:
        windows = [7, 30]

    result = daily_counts.copy()

    # Для каждого тикера вычисляем rolling отдельно
    for ticker in result['ticker'].unique():
        ticker_mask = result['ticker'] == ticker
        ticker_data = result[ticker_mask].sort_values('date')

        for window in windows:
            col_name = f'news_count_{window}d'
            result.loc[ticker_mask, col_name] = ticker_data['news_count_1d'].rolling(
                window=window,
                min_periods=1
            ).sum().values

    # Заполнить NaN нулями
    for window in windows:
        col_name = f'news_count_{window}d'
        if col_name in result.columns:
            result[col_name] = result[col_name].fillna(0).astype(int)

    return result


def add_llm_sentiment_features(
    ticker_sentiment_df: pd.DataFrame,
    rolling_windows: list[int] = None
) -> pd.DataFrame:
    """
    Агрегация LLM sentiment на уровне (date, ticker)

    Args:
        ticker_sentiment_df: DataFrame с колонками [publish_date, ticker, sentiment, confidence]
        rolling_windows: окна для rolling features (по умолчанию [7, 30])

    Returns:
        DataFrame с признаками:
        - sentiment_mean: средний sentiment (-1 до 1)
        - sentiment_weighted: sentiment взвешенный по confidence
        - confidence_mean: средняя уверенность (0-10)
        - positive_count: количество позитивных новостей
        - negative_count: количество негативных новостей
        - neutral_count: количество нейтральных новостей
        + rolling features для всех метрик

    Example:
        >>> ticker_sentiment = pd.read_csv('news_ticker_sentiment.csv')
        >>> features = add_llm_sentiment_features(ticker_sentiment)
    """
    if rolling_windows is None:
        rolling_windows = [7, 30]

    df = ticker_sentiment_df.copy()

    # Проверка наличия sentiment колонок
    if 'sentiment' not in df.columns:
        raise ValueError("DataFrame must contain 'sentiment' column")

    # Преобразовать дату
    df['publish_date'] = pd.to_datetime(df['publish_date'])
    df['date'] = df['publish_date'].dt.date
    df['date'] = pd.to_datetime(df['date'])

    # Группировка по (date, ticker) с агрегацией
    grouped = df.groupby(['date', 'ticker']).agg(
        sentiment_mean=('sentiment', 'mean'),
        positive_count=('sentiment', lambda x: (x == 1).sum()),
        negative_count=('sentiment', lambda x: (x == -1).sum()),
        neutral_count=('sentiment', lambda x: (x == 0).sum()),
        confidence_mean=('confidence', 'mean'),
        # Weighted sentiment: sum(sentiment * confidence) / sum(confidence)
        sentiment_sum=('sentiment', 'sum'),
        confidence_sum=('confidence', 'sum')
    ).reset_index()

    # Вычисляем weighted sentiment
    # Избегаем деления на ноль
    mask = grouped['confidence_sum'] > 0
    grouped['sentiment_weighted'] = 0.0
    grouped.loc[mask, 'sentiment_weighted'] = (
        df.groupby(['date', 'ticker']).apply(
            lambda g: (g['sentiment'] * g['confidence']).sum() / g['confidence'].sum()
            if g['confidence'].sum() > 0 else 0,
            include_groups=False
        ).loc[grouped.loc[mask, ['date', 'ticker']].apply(tuple, axis=1)].values
    )

    # Убираем вспомогательные колонки
    grouped = grouped.drop(columns=['sentiment_sum', 'confidence_sum'])

    # Добавляем rolling features для каждого тикера
    result = grouped.copy()

    for ticker in result['ticker'].unique():
        ticker_mask = result['ticker'] == ticker
        ticker_data = result[ticker_mask].sort_values('date')

        # Rolling для каждой метрики
        for window in rolling_windows:
            # sentiment_mean rolling
            col_name = f'sentiment_mean_{window}d'
            result.loc[ticker_mask, col_name] = ticker_data['sentiment_mean'].rolling(
                window=window, min_periods=1
            ).mean().values

            # confidence_mean rolling
            col_name = f'confidence_mean_{window}d'
            result.loc[ticker_mask, col_name] = ticker_data['confidence_mean'].rolling(
                window=window, min_periods=1
            ).mean().values

            # positive_count rolling
            col_name = f'positive_count_{window}d'
            result.loc[ticker_mask, col_name] = ticker_data['positive_count'].rolling(
                window=window, min_periods=1
            ).sum().values

            # negative_count rolling
            col_name = f'negative_count_{window}d'
            result.loc[ticker_mask, col_name] = ticker_data['negative_count'].rolling(
                window=window, min_periods=1
            ).sum().values

    # Заполнить NaN нулями для count колонок
    count_cols = [col for col in result.columns if 'count' in col]
    for col in count_cols:
        result[col] = result[col].fillna(0).astype(int)

    return result


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
        rolling_windows: окна для rolling features (по умолчанию [7, 30])
        include_llm_features: добавить LLM признаки (sentiment, confidence)

    Returns:
        DataFrame с признаками на уровне (date, ticker):
        - date: дата
        - ticker: тикер
        - news_count_1d: количество новостей за день
        - news_count_7d: количество новостей за 7 дней
        - news_count_30d: количество новостей за 30 дней
        - [опционально] sentiment_mean, sentiment_weighted, confidence_mean, positive_count, etc.

    Example:
        >>> # Базовые features (только counts)
        >>> ticker_sentiment = pd.read_csv('news_ticker_sentiment.csv')
        >>> features = aggregate_ticker_news_features(ticker_sentiment, include_llm_features=False)
        >>> features.head()
           date       ticker  news_count_1d  news_count_7d  news_count_30d
        0  2020-01-01  SBER   5             23            150

        >>> # С LLM features
        >>> features_llm = aggregate_ticker_news_features(ticker_sentiment, include_llm_features=True)
        >>> features_llm.head()
           date       ticker  news_count_1d  sentiment_mean  confidence_mean  ...
        0  2020-01-01  SBER   5             0.6            7.2              ...
    """
    if rolling_windows is None:
        rolling_windows = [7, 30]

    # 1. Подсчет новостей по дням (базовые counts)
    daily_counts = compute_ticker_daily_counts(ticker_sentiment_df)

    # 2. Rolling features для counts
    features = add_rolling_features(daily_counts, windows=rolling_windows)

    # 3. LLM features (если доступны)
    if include_llm_features and 'sentiment' in ticker_sentiment_df.columns:
        # Вычисляем LLM sentiment features
        llm_features = add_llm_sentiment_features(ticker_sentiment_df, rolling_windows=rolling_windows)

        # Merge с базовыми features
        features = features.merge(
            llm_features,
            on=['date', 'ticker'],
            how='left'
        )

    # Сортировка для удобства
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
    news_cols = ['ticker', 'date_shifted'] + [
        col for col in news.columns
        if col.startswith('news_count_')
        or col.startswith('sentiment_')
        or col.startswith('confidence_')
        or col.startswith('positive_count')
        or col.startswith('negative_count')
        or col.startswith('neutral_count')
        or col.startswith('topic_')
    ]

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
        if col.startswith('news_count_')
        or col.startswith('positive_count')
        or col.startswith('negative_count')
        or col.startswith('neutral_count')
    ]
    for col in count_cols:
        candles[col] = candles[col].fillna(0).astype(int)

    # Для sentiment/confidence заполняем 0 (нейтральный sentiment если нет новостей)
    sentiment_cols = [
        col for col in candles.columns
        if col.startswith('sentiment_')
        or col.startswith('confidence_')
    ]
    for col in sentiment_cols:
        candles[col] = candles[col].fillna(0.0)

    return candles
