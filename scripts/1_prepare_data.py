"""
Data Preparation Script

Этот скрипт:
1. Загружает train_candles.csv
2. Создает признаки (momentum, volatility, MA, RSI, MACD, etc.)
3. Разбивает на train/val/test (70%/15%/15% по времени)
4. Обрабатывает public_test и private_test с теми же фичами
5. Сохраняет в data/preprocessed/*.parquet

Usage:
    python scripts/1_prepare_data.py
    python scripts/1_prepare_data.py --train-ratio 0.7 --val-ratio 0.15
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime
from typing import Optional

# Добавляем src/ в PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import pandas as pd

from finam.features import (
    add_all_features,
    fit_cross_sectional_stats,
    transform_cross_sectional_features
)
from finam.features_news_tickers import join_ticker_news_features
from finam.features_target import compute_multi_horizon_targets


def prepare_data(
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    windows: list = None,
    include_cross_sectional: bool = True,
    include_interactions: bool = True,
    start_date: Optional[str] = None
):
    """
    Подготовка данных с разбиением на train/val/test

    Args:
        train_ratio: доля train (например 0.7 = 70%)
        val_ratio: доля validation (например 0.15 = 15%)
        windows: окна для признаков [5, 20]
        include_cross_sectional: включить cross-sectional features
        include_interactions: включить interaction features
        start_date: earliest candle date to keep (формат YYYY-MM-DD)
    """
    print("=" * 80)
    print("DATA PREPARATION PIPELINE")
    print("=" * 80 + "\n")

    if windows is None:
        windows = [5, 20]

    start_ts = pd.to_datetime(start_date) if start_date else None

    # Проверка что train + val <= 1.0
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio < 0 or test_ratio > 1.0:
        raise ValueError(f"Invalid ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")

    print(f"Split ratios: train={train_ratio:.1%}, val={val_ratio:.1%}, test={test_ratio:.1%}\n")

    # ========================================================================
    # 1. Загрузка данных
    # ========================================================================
    print("[1/5] Loading data...")

    data_dir = project_root / 'data' / 'raw' / 'participants'
    train_path = data_dir / 'candles.csv'  # основные данные для обучения
    news_path = data_dir / 'news.csv'
    holdout_test_path = data_dir / 'candles_2.csv'  # финальный holdout test
    test_news_path = data_dir / 'news_2.csv'

    if not train_path.exists():
        print(f"❌ File not found: {train_path}")
        return

    # Загружаем свечи
    df = pd.read_csv(train_path)
    df['begin'] = pd.to_datetime(df['begin'])

    if start_ts is not None:
        before_len = len(df)
        df = df[df['begin'] >= start_ts].copy()
        print(
            f"   * Applied start_date filter ({start_ts.date()}), "
            f"dropped {before_len - len(df)} rows"
        )

    print(f"   OK Loaded {len(df)} rows, {df['ticker'].nunique()} tickers (train)")
    print(f"   OK Period: {df['begin'].min()} to {df['begin'].max()}")

    # Загружаем ticker news features (из preprocessed_news/)
    preprocessed_news_dir = project_root / 'data' / 'preprocessed_news'
    ticker_news_path = preprocessed_news_dir / 'news_ticker_features.csv'
    ticker_news_df = None
    if ticker_news_path.exists():
        ticker_news_df = pd.read_csv(ticker_news_path)
        # Проверяем наличие LLM features
        has_llm = any(col.startswith('sentiment_') for col in ticker_news_df.columns)
        feature_type = "with LLM sentiment" if has_llm else "counts only"
        print(f"   OK Loaded {len(ticker_news_df)} ticker news features (train, {feature_type})")
    else:
        print(f"   [WARNING] Ticker news features not found: {ticker_news_path}")
        print("   [INFO] Run: python scripts/0_1_news_ticker_features.py")

    # Загружаем holdout test данные
    holdout_test_df = None
    test_news_df = None

    if holdout_test_path.exists():
        holdout_test_df = pd.read_csv(holdout_test_path)
        holdout_test_df['begin'] = pd.to_datetime(holdout_test_df['begin'])
        print(f"   OK Loaded {len(holdout_test_df)} rows (holdout_test)")
        print(f"   OK Period: {holdout_test_df['begin'].min()} to {holdout_test_df['begin'].max()}")

    ticker_test_news_path = preprocessed_news_dir / 'news_2_ticker_features.csv'
    if ticker_test_news_path.exists():
        test_news_df = pd.read_csv(ticker_test_news_path)
        # Проверяем наличие LLM features
        has_llm_test = any(col.startswith('sentiment_') for col in test_news_df.columns)
        feature_type_test = "with LLM sentiment" if has_llm_test else "counts only"
        print(f"   OK Loaded {len(test_news_df)} ticker news features (holdout_test, {feature_type_test})")

    print()

    # ========================================================================
    # 2. Feature Engineering (базовые фичи БЕЗ cross-sectional)
    # ========================================================================
    print("[2/5] Feature Engineering (базовые фичи)...")

    # Создаём базовые фичи (momentum, volatility, MA, RSI, MACD, volume)
    # НО БЕЗ cross-sectional (ranks, z-scores), чтобы избежать data leakage
    df = add_all_features(
        df,
        windows=windows,
        include_cross_sectional=False,  # ⚠️ ВАЖНО: сначала без cross-sectional
        include_interactions=include_interactions
    )

    # Добавляем ticker news features (если загружены)
    if ticker_news_df is not None:
        print("   * Joining ticker news features...")
        df = join_ticker_news_features(df, ticker_news_df, lag_days=1)
        print(f"      OK Added {len([c for c in df.columns if c.startswith('news_count_')])} news features")

    print("   OK Created базовые признаки")

    # Вычисляем таргеты для всех 20 горизонтов
    print("   * Computing 20 target returns...")
    df = compute_multi_horizon_targets(df, horizons=list(range(1, 21)))
    print("   OK Added targets: target_return_1d through target_return_20d")

    # ========================================================================
    # 3. Train/Val/Test Split (temporal)
    # ========================================================================
    print("[3/5] Creating train/val/test split...")

    # Сортируем по дате
    df = df.sort_values('begin').reset_index(drop=True)

    # Временной split
    unique_dates = sorted(df['begin'].unique())
    n_dates = len(unique_dates)

    train_end_idx = int(n_dates * train_ratio)
    val_end_idx = int(n_dates * (train_ratio + val_ratio))

    train_end_date = unique_dates[train_end_idx]
    val_end_date = unique_dates[val_end_idx]

    train_df = df[df['begin'] < train_end_date].copy()
    val_df = df[(df['begin'] >= train_end_date) & (df['begin'] < val_end_date)].copy()
    test_df = df[df['begin'] >= val_end_date].copy()

    # Удаляем строки с NaN хотя бы в одном из таргетов
    # (для длинных горизонтов в конце периода может не быть таргетов)
    target_cols = [f'target_return_{h}d' for h in range(1, 21)]

    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        original_len = len(split_df)
        # Удаляем только если ВСЕ таргеты NaN (для коротких горизонтов может быть таргет)
        split_df.dropna(subset=target_cols, how='all', inplace=True)
        removed = original_len - len(split_df)
        if removed > 0:
            print(f"   [INFO] Removed {removed} rows with all NaN targets from {split_name}")

    print(f"\n   Train: {len(train_df):5d} rows ({train_df['begin'].min().date()} to {train_df['begin'].max().date()})")
    print(f"   Val:   {len(val_df):5d} rows ({val_df['begin'].min().date()} to {val_df['begin'].max().date()})")
    print(f"   Test:  {len(test_df):5d} rows ({test_df['begin'].min().date()} to {test_df['begin'].max().date()})")
    print("\n   Split dates:")
    print(f"      train_end: {train_end_date.date()}")
    print(f"      val_end:   {val_end_date.date()}\n")

    # ========================================================================
    # 3.5. Cross-Sectional Features (БЕЗ data leakage!)
    # ========================================================================
    if include_cross_sectional:
        print("[3.5/5] Cross-sectional features (без data leakage)...")

        # Шаг 1: Фитим статистики ТОЛЬКО на train
        print("   * Fitting cross-sectional stats on train data...")
        cross_sectional_stats = fit_cross_sectional_stats(train_df)

        # Шаг 2: Применяем к train (fit_mode=True)
        print("   * Transforming train data...")
        train_df = transform_cross_sectional_features(train_df, fit_mode=True)

        # Шаг 3: Применяем к val/test (используя train статистики)
        print("   * Transforming val data (using train stats)...")
        val_df = transform_cross_sectional_features(val_df, stats=cross_sectional_stats)

        print("   * Transforming test data (using train stats)...")
        test_df = transform_cross_sectional_features(test_df, stats=cross_sectional_stats)

        print("   OK Cross-sectional features добавлены БЕЗ data leakage!\n")

    # Получаем список feature columns (все кроме основных и таргетов)
    base_cols = ['ticker', 'begin', 'open', 'high', 'low', 'close', 'volume', 'adj_close']
    target_cols = [col for col in train_df.columns if col.startswith('target_')]
    feature_cols = [col for col in train_df.columns if col not in base_cols and col not in target_cols]

    print(f"   ИТОГО: {len(feature_cols)} features")
    print("\n   Sample features (first 10):")
    for i, col in enumerate(feature_cols[:10], 1):
        print(f"      {i}. {col}")
    if len(feature_cols) > 10:
        print(f"      ... and {len(feature_cols) - 10} more\n")

    # ========================================================================
    # 4. Feature Engineering (holdout test data)
    # ========================================================================
    print("[4/5] Feature Engineering (holdout test data)...")

    # Обрабатываем holdout_test
    if holdout_test_df is not None:
        # Базовые фичи БЕЗ cross-sectional
        holdout_test_df = add_all_features(
            holdout_test_df,
            windows=windows,
            include_cross_sectional=False,
            include_interactions=include_interactions
        )
        if test_news_df is not None:
            print("   * Joining ticker news features (holdout test)...")
            holdout_test_df = join_ticker_news_features(holdout_test_df, test_news_df, lag_days=1)

        # Cross-sectional features (используя train статистики)
        if include_cross_sectional:
            holdout_test_df = transform_cross_sectional_features(holdout_test_df, stats=cross_sectional_stats)

        print(f"   OK Processed holdout_test: {len(holdout_test_df)} rows, {len(feature_cols)} features")
    else:
        print("   [WARNING] No holdout test data found")

    print()

    # ========================================================================
    # 5. Сохранение
    # ========================================================================
    print("[5/5] Saving preprocessed data...")

    output_dir = project_root / 'data' / 'preprocessed'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Сохраняем train/val/test splits
    train_df.to_csv(output_dir / 'train.csv', index=False)
    val_df.to_csv(output_dir / 'val.csv', index=False)
    test_df.to_csv(output_dir / 'test.csv', index=False)

    print(f"   OK Saved to {output_dir}/")
    print(f"      - train.csv ({len(train_df)} rows)")
    print(f"      - val.csv ({len(val_df)} rows)")
    print(f"      - test.csv ({len(test_df)} rows)")

    # Сохраняем holdout test
    if holdout_test_df is not None:
        holdout_test_df.to_csv(output_dir / 'holdout_test.csv', index=False)
        print(f"      - holdout_test.csv ({len(holdout_test_df)} rows)")

    # Сохраняем metadata
    metadata = {
        'created_at': datetime.now().isoformat(),
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio,
        'windows': windows,
        'include_cross_sectional': include_cross_sectional,
        'include_interactions': include_interactions,
        'n_features': len(feature_cols),
        'feature_columns': feature_cols,
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'train_period': f"{train_df['begin'].min().date()} to {train_df['begin'].max().date()}",
        'val_period': f"{val_df['begin'].min().date()} to {val_df['begin'].max().date()}",
        'test_period': f"{test_df['begin'].min().date()} to {test_df['begin'].max().date()}",
        'train_end_date': str(train_end_date.date()),
        'val_end_date': str(val_end_date.date()),
        'start_date': start_date
    }

    # Добавляем информацию о holdout test если есть
    if holdout_test_df is not None:
        metadata['holdout_test_size'] = len(holdout_test_df)
        metadata['holdout_test_period'] = f"{holdout_test_df['begin'].min().date()} to {holdout_test_df['begin'].max().date()}"

    import json
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("   OK Saved metadata.json\n")

    # ========================================================================
    # Summary
    # ========================================================================
    print("=" * 80)
    print("DATA PREPARATION COMPLETE!")
    print("=" * 80 + "\n")

    print("Summary:")
    print(f"   Total rows:     {len(df)}")
    print(f"   Features:       {len(feature_cols)}")
    print(f"   Train size:     {len(train_df)} ({train_ratio:.1%})")
    print(f"   Val size:       {len(val_df)} ({val_ratio:.1%})")
    print(f"   Test size:      {len(test_df)} ({test_ratio:.1%})")

    if holdout_test_df is not None:
        print(f"   Holdout test:   {len(holdout_test_df)} rows")

    print(f"\n   Saved to: {output_dir}/")
    print("\nNext steps:")
    print("   # Train a new model")
    print("   python scripts/2_train_model.py --exp-name my_experiment")
    print("\n   # Or compute baseline metrics")
    print("   python scripts/compute_baseline_metrics.py")
    print("\n   # After training, generate submission")
    print("   python scripts/4_generate_submission.py --run-id <timestamp>_<exp_name>")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare data with train/val/test split')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='Train ratio (default: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                        help='Validation ratio (default: 0.15)')
    parser.add_argument('--windows', type=int, nargs='+', default=[5, 20],
                        help='Feature windows (default: 5 20)')
    parser.add_argument('--no-cross-sectional', action='store_true',
                        help='Exclude cross-sectional features')
    parser.add_argument('--no-interactions', action='store_true',
                        help='Exclude interaction features')
    parser.add_argument('--start-date', type=str, default=None,
                        help='Earliest candle date to keep (YYYY-MM-DD)')

    args = parser.parse_args()

    prepare_data(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        windows=args.windows,
        include_cross_sectional=not args.no_cross_sectional,
        include_interactions=not args.no_interactions,
        start_date=args.start_date
    )
