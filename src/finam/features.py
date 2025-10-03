"""
Feature Engineering для FORECAST задачи

Базируется на scripts/baseline_solution.py:60-96, но добавляет:
- Advanced technical indicators (RSI, MACD, Bollinger Bands, ATR)
- Cross-sectional features (ranks, z-scores)
- Volume features
- Interaction features

ВАЖНО: все features используют ТОЛЬКО прошлые данные (no look-ahead bias!)
"""

import numpy as np
import pandas as pd
from scipy.stats import zscore


def compute_momentum(
    df: pd.DataFrame,
    windows: list[int] = [5, 20]
) -> pd.DataFrame:
    """
    Моментум = процентное изменение цены за N дней

    Args:
        df: DataFrame с колонками ['ticker', 'begin', 'close']
        windows: список размеров окон (например [5, 20])

    Returns:
        df с добавленными колонками momentum_{window}d

    Example:
        momentum_5d[t] = (close[t] - close[t-5]) / close[t-5]
    """
    df = df.copy()

    for window in windows:
        col_name = f'momentum_{window}d'
        df[col_name] = df.groupby('ticker')['close'].pct_change(window)

    return df


def compute_volatility(
    df: pd.DataFrame,
    windows: list[int] = [5, 20]
) -> pd.DataFrame:
    """
    Волатильность = std дневных доходностей за N дней

    Также добавляет Garman-Klass volatility (использует high/low)

    Args:
        df: DataFrame с колонками ['ticker', 'high', 'low', 'open', 'close']
        windows: список размеров окон

    Returns:
        df с добавленными колонками volatility_{window}d, gk_volatility

    Example:
        volatility_5d[t] = std(daily_returns[t-4:t+1])
    """
    df = df.copy()

    # Close-to-close volatility
    daily_returns = df.groupby('ticker')['close'].pct_change()

    for window in windows:
        col_name = f'volatility_{window}d'
        df[col_name] = daily_returns.groupby(df['ticker']).rolling(window).std().reset_index(level=0, drop=True)

    # Garman-Klass volatility (более точная, использует intraday range)
    # Formula: sqrt(0.5 * ln(H/L)^2 - (2*ln(2) - 1) * ln(C/O)^2)
    log_hl = np.log(df['high'] / df['low'])
    log_co = np.log(df['close'] / df['open'])

    gk_variance = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    df['gk_volatility'] = np.sqrt(gk_variance.clip(lower=0))

    return df


def compute_moving_averages(
    df: pd.DataFrame,
    windows: list[int] = [5, 20]
) -> pd.DataFrame:
    """
    Скользящие средние и расстояние от них

    Args:
        df: DataFrame с колонкой 'close'
        windows: список размеров окон

    Returns:
        df с добавленными колонками ma_{window}d, distance_from_ma_{window}d

    Example:
        ma_5d[t] = mean(close[t-4:t+1])
        distance_from_ma_5d[t] = (close[t] - ma_5d[t]) / ma_5d[t]
    """
    df = df.copy()

    for window in windows:
        ma_col = f'ma_{window}d'
        dist_col = f'distance_from_ma_{window}d'

        # MA
        df[ma_col] = df.groupby('ticker')['close'].rolling(window).mean().reset_index(level=0, drop=True)

        # Distance from MA (normalized)
        df[dist_col] = (df['close'] - df[ma_col]) / df[ma_col]

    return df


def compute_rsi(
    df: pd.DataFrame,
    window: int = 14
) -> pd.DataFrame:
    """
    Relative Strength Index (RSI)

    RSI = 100 - (100 / (1 + RS))
    где RS = average_gain / average_loss за N дней

    Args:
        df: DataFrame с колонкой 'close'
        window: размер окна (обычно 14)

    Returns:
        df с колонкой rsi_{window}d
    """
    df = df.copy()

    # Дневные изменения цены
    delta = df.groupby('ticker')['close'].diff()

    # Разделяем на gains и losses
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # EWM для сглаживания (как в классическом RSI)
    avg_gain = gain.groupby(df['ticker']).rolling(window).mean().reset_index(level=0, drop=True)
    avg_loss = loss.groupby(df['ticker']).rolling(window).mean().reset_index(level=0, drop=True)

    # RS и RSI
    rs = avg_gain / (avg_loss + 1e-10)  # +epsilon для избежания деления на ноль
    df[f'rsi_{window}d'] = 100 - (100 / (1 + rs))

    return df


def compute_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> pd.DataFrame:
    """
    MACD (Moving Average Convergence Divergence)

    MACD = EMA_fast - EMA_slow
    Signal = EMA(MACD, signal_period)
    Histogram = MACD - Signal

    Args:
        df: DataFrame с колонкой 'close'
        fast: период быстрой EMA (обычно 12)
        slow: период медленной EMA (обычно 26)
        signal: период signal line (обычно 9)

    Returns:
        df с колонками macd, macd_signal, macd_diff
    """
    df = df.copy()

    # EMA (exponential moving average)
    ema_fast = df.groupby('ticker')['close'].ewm(span=fast, adjust=False).mean().reset_index(level=0, drop=True)
    ema_slow = df.groupby('ticker')['close'].ewm(span=slow, adjust=False).mean().reset_index(level=0, drop=True)

    # MACD line
    df['macd'] = ema_fast - ema_slow

    # Signal line
    df['macd_signal'] = df.groupby('ticker')['macd'].ewm(span=signal, adjust=False).mean().reset_index(level=0, drop=True)

    # Histogram
    df['macd_diff'] = df['macd'] - df['macd_signal']

    return df


def compute_bollinger_bands(
    df: pd.DataFrame,
    window: int = 20,
    num_std: float = 2.0
) -> pd.DataFrame:
    """
    Bollinger Bands

    Middle Band = SMA(close, window)
    Upper Band = Middle + num_std * std
    Lower Band = Middle - num_std * std

    Также считаем позицию цены относительно band:
    bb_position = (close - lower) / (upper - lower)

    Args:
        df: DataFrame с колонкой 'close'
        window: размер окна (обычно 20)
        num_std: количество стандартных отклонений (обычно 2)

    Returns:
        df с колонками bb_upper, bb_middle, bb_lower, bb_position_{window}d
    """
    df = df.copy()

    # Middle band = SMA
    bb_middle = df.groupby('ticker')['close'].rolling(window).mean().reset_index(level=0, drop=True)

    # Std
    bb_std = df.groupby('ticker')['close'].rolling(window).std().reset_index(level=0, drop=True)

    # Upper/Lower bands
    df['bb_upper'] = bb_middle + num_std * bb_std
    df['bb_middle'] = bb_middle
    df['bb_lower'] = bb_middle - num_std * bb_std

    # Position within bands [0, 1]
    # 0 = at lower band, 0.5 = at middle, 1 = at upper band
    band_width = df['bb_upper'] - df['bb_lower']
    df[f'bb_position_{window}d'] = (df['close'] - df['bb_lower']) / (band_width + 1e-10)

    # Cleanup temporary columns
    df = df.drop(columns=['bb_upper', 'bb_middle', 'bb_lower'])

    return df


def compute_volume_features(
    df: pd.DataFrame,
    windows: list[int] = [5, 20]
) -> pd.DataFrame:
    """
    Volume features

    - log_volume: log(1 + volume) для нормализации
    - volume_ratio_{window}d: volume / mean(volume) за последние N дней

    Args:
        df: DataFrame с колонкой 'volume'
        windows: список размеров окон

    Returns:
        df с добавленными колонками
    """
    df = df.copy()

    # Log volume
    df['log_volume'] = np.log1p(df['volume'])

    # Volume ratio (текущий объём / средний объём)
    for window in windows:
        col_name = f'volume_ratio_{window}d'
        avg_volume = df.groupby('ticker')['volume'].rolling(window).mean().reset_index(level=0, drop=True)
        df[col_name] = df['volume'] / (avg_volume + 1e-10)

    return df


def compute_cross_sectional_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional features — ранжирование и z-scores по дате

    Для каждого признака создаём:
    - {feature}_rank: процентильный rank по дате (0-1)
    - {feature}_zscore: z-score по дате

    Это отвечает на вопрос: "Насколько этот актив выделяется на фоне других в этот день?"

    Args:
        df: DataFrame с колонкой 'begin' (дата)

    Returns:
        df с добавленными rank и zscore колонками

    Example:
        Если у актива momentum_5d_rank = 0.9, значит он в топ-10% по momentum в этот день
    """
    df = df.copy()

    # Признаки для которых создаём cross-sectional версии
    features_to_rank = [
        'momentum_5d', 'momentum_20d',
        'volatility_5d', 'volatility_20d',
        'distance_from_ma_5d', 'distance_from_ma_20d',
        'rsi_14d', 'macd_diff',
        'volume_ratio_5d', 'volume_ratio_20d'
    ]

    # Проверяем какие признаки есть в df
    features_to_rank = [f for f in features_to_rank if f in df.columns]

    for feature in features_to_rank:
        # Percentile rank (0-1)
        rank_col = f'{feature}_rank'
        df[rank_col] = df.groupby('begin')[feature].rank(pct=True)

        # Z-score (по дате)
        zscore_col = f'{feature}_zscore'
        df[zscore_col] = df.groupby('begin')[feature].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-10)
        )

    return df


def compute_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interaction features — комбинации признаков

    Примеры:
    - momentum × volatility (высокий momentum при низкой vol = более надёжный сигнал)
    - volume_ratio × price_change (всплеск объёма при росте = сильный сигнал)

    Args:
        df: DataFrame с уже вычисленными features

    Returns:
        df с добавленными interaction features
    """
    df = df.copy()

    # momentum × volatility
    if 'momentum_5d' in df.columns and 'volatility_5d' in df.columns:
        df['momentum_vol_5d'] = df['momentum_5d'] * (1 / (df['volatility_5d'] + 1e-10))

    if 'momentum_20d' in df.columns and 'volatility_20d' in df.columns:
        df['momentum_vol_20d'] = df['momentum_20d'] * (1 / (df['volatility_20d'] + 1e-10))

    # volume_ratio × price_change
    if 'volume_ratio_5d' in df.columns and 'momentum_5d' in df.columns:
        df['volume_momentum_5d'] = df['volume_ratio_5d'] * df['momentum_5d']

    return df


def add_all_features(
    df: pd.DataFrame,
    windows: list[int] = [5, 20],
    include_cross_sectional: bool = True,
    include_interactions: bool = True
) -> pd.DataFrame:
    """
    Добавить ВСЕ features к DataFrame

    Это главная функция для feature engineering!

    Args:
        df: DataFrame с OHLCV данными
        windows: список размеров окон для rolling features
        include_cross_sectional: добавить cross-sectional features (ranks, z-scores)
        include_interactions: добавить interaction features

    Returns:
        df с добавленными features

    Example:
        >>> df = pd.read_csv('train_candles.csv')
        >>> df = add_all_features(df, windows=[5, 20])
        >>> # Теперь df содержит ~50+ признаков для ML модели
    """
    print("Feature Engineering...")

    # Ensure sorted by ticker and date
    df = df.sort_values(['ticker', 'begin']).reset_index(drop=True)

    # 1. Базовые features
    print("   * Momentum...")
    df = compute_momentum(df, windows=windows)

    print("   * Volatility...")
    df = compute_volatility(df, windows=windows)

    print("   * Moving Averages...")
    df = compute_moving_averages(df, windows=windows)

    # 2. Technical indicators
    print("   * RSI...")
    df = compute_rsi(df, window=14)

    print("   * MACD...")
    df = compute_macd(df)

    print("   * Bollinger Bands...")
    df = compute_bollinger_bands(df, window=20)

    # 3. Volume features
    print("   * Volume features...")
    df = compute_volume_features(df, windows=windows)

    # 4. Cross-sectional features
    if include_cross_sectional:
        print("   * Cross-sectional (ranks, z-scores)...")
        df = compute_cross_sectional_features(df)

    # 5. Interaction features
    if include_interactions:
        print("   * Interaction features...")
        df = compute_interaction_features(df)

    print(f"   OK Total features: {len(df.columns)}")

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """
    Получить список колонок с features (исключая OHLCV, targets, metadata)

    Args:
        df: DataFrame после add_all_features

    Returns:
        список названий колонок с features
    """
    # Колонки которые НЕ являются features
    exclude_cols = [
        'ticker', 'begin', 'dataset',
        'open', 'high', 'low', 'close', 'volume',
        'target_return_1d', 'target_direction_1d',
        'target_return_20d', 'target_direction_20d',
        'pred_return_1d', 'pred_return_20d',
        'pred_prob_up_1d', 'pred_prob_up_20d'
    ]

    feature_cols = [col for col in df.columns if col not in exclude_cols]

    return feature_cols
