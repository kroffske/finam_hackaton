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

# Добавляем src/ в PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import pandas as pd
import numpy as np

from finam.features import add_all_features
from finam.features_news import add_news_features


def prepare_data(
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    windows: list = None,
    include_cross_sectional: bool = True,
    include_interactions: bool = True
):
    """
    Подготовка данных с разбиением на train/val/test

    Args:
        train_ratio: доля train (например 0.7 = 70%)
        val_ratio: доля validation (например 0.15 = 15%)
        windows: окна для признаков [5, 20]
        include_cross_sectional: включить cross-sectional features
        include_interactions: включить interaction features
    """
    print("=" * 80)
    print("DATA PREPARATION PIPELINE")
    print("=" * 80 + "\n")

    if windows is None:
        windows = [5, 20]

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
    train_path = data_dir / 'train_candles.csv'
    news_path = data_dir / 'train_news.csv'
    public_test_path = data_dir / 'public_test_candles.csv'
    private_test_path = data_dir / 'private_test_candles.csv'
    test_news_path = data_dir / 'test_news.csv'

    if not train_path.exists():
        print(f"❌ File not found: {train_path}")
        return

    # Загружаем свечи
    df = pd.read_csv(train_path)
    df['begin'] = pd.to_datetime(df['begin'])

    print(f"   OK Loaded {len(df)} rows, {df['ticker'].nunique()} tickers (train)")
    print(f"   OK Period: {df['begin'].min()} to {df['begin'].max()}")

    # Загружаем новости (если файл существует)
    news_df = None
    if news_path.exists():
        news_df = pd.read_csv(news_path)
        print(f"   OK Loaded {len(news_df)} news (train)")
    else:
        print(f"   [WARNING] News file not found: {news_path}")

    # Загружаем test данные
    public_test_df = None
    private_test_df = None
    test_news_df = None

    if public_test_path.exists():
        public_test_df = pd.read_csv(public_test_path)
        public_test_df['begin'] = pd.to_datetime(public_test_df['begin'])
        print(f"   OK Loaded {len(public_test_df)} rows (public_test)")

    if private_test_path.exists():
        private_test_df = pd.read_csv(private_test_path)
        private_test_df['begin'] = pd.to_datetime(private_test_df['begin'])
        print(f"   OK Loaded {len(private_test_df)} rows (private_test)")

    if test_news_path.exists():
        test_news_df = pd.read_csv(test_news_path)
        print(f"   OK Loaded {len(test_news_df)} news (test)")

    print()

    # ========================================================================
    # 2. Feature Engineering (train data)
    # ========================================================================
    print("[2/5] Feature Engineering (train data)...")

    df = add_all_features(
        df,
        windows=windows,
        include_cross_sectional=include_cross_sectional,
        include_interactions=include_interactions
    )

    # Добавляем новостные фичи (если новости загружены)
    if news_df is not None:
        df = add_news_features(df, news_df, lag_days=1, rolling_windows=[1, 7, 30])

    # Получаем список feature columns (все кроме основных и таргетов)
    base_cols = ['ticker', 'begin', 'open', 'high', 'low', 'close', 'volume', 'adj_close']
    target_cols = [col for col in df.columns if col.startswith('target_')]
    feature_cols = [col for col in df.columns if col not in base_cols and col not in target_cols]

    print(f"   OK Created {len(feature_cols)} features for train data")
    print(f"\n   Sample features (first 10):")
    for i, col in enumerate(feature_cols[:10], 1):
        print(f"      {i}. {col}")
    if len(feature_cols) > 10:
        print(f"      ... and {len(feature_cols) - 10} more\n")

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

    # Удаляем строки с NaN в таргетах
    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        original_len = len(split_df)
        split_df.dropna(subset=['target_return_1d', 'target_return_20d'], inplace=True)
        removed = original_len - len(split_df)
        if removed > 0:
            print(f"   [INFO] Removed {removed} rows with NaN targets from {split_name}")

    print(f"\n   Train: {len(train_df):5d} rows ({train_df['begin'].min().date()} to {train_df['begin'].max().date()})")
    print(f"   Val:   {len(val_df):5d} rows ({val_df['begin'].min().date()} to {val_df['begin'].max().date()})")
    print(f"   Test:  {len(test_df):5d} rows ({test_df['begin'].min().date()} to {test_df['begin'].max().date()})")
    print(f"\n   Split dates:")
    print(f"      train_end: {train_end_date.date()}")
    print(f"      val_end:   {val_end_date.date()}\n")

    # ========================================================================
    # 4. Feature Engineering (public/private test data)
    # ========================================================================
    print("[4/5] Feature Engineering (public/private test data)...")

    # Обрабатываем public_test
    if public_test_df is not None:
        public_test_df = add_all_features(
            public_test_df,
            windows=windows,
            include_cross_sectional=include_cross_sectional,
            include_interactions=include_interactions
        )
        if test_news_df is not None:
            public_test_df = add_news_features(public_test_df, test_news_df, lag_days=1, rolling_windows=[1, 7, 30])
        print(f"   OK Processed public_test: {len(public_test_df)} rows, {len(feature_cols)} features")

    # Обрабатываем private_test
    if private_test_df is not None:
        private_test_df = add_all_features(
            private_test_df,
            windows=windows,
            include_cross_sectional=include_cross_sectional,
            include_interactions=include_interactions
        )
        if test_news_df is not None:
            private_test_df = add_news_features(private_test_df, test_news_df, lag_days=1, rolling_windows=[1, 7, 30])
        print(f"   OK Processed private_test: {len(private_test_df)} rows, {len(feature_cols)} features")

    print()

    # ========================================================================
    # 5. Сохранение
    # ========================================================================
    print("[5/5] Saving preprocessed data...")

    output_dir = project_root / 'data' / 'preprocessed'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Сохраняем train/val/test splits
    train_df.to_parquet(output_dir / 'train.parquet', index=False)
    val_df.to_parquet(output_dir / 'val.parquet', index=False)
    test_df.to_parquet(output_dir / 'test.parquet', index=False)

    print(f"   OK Saved to {output_dir}/")
    print(f"      - train.parquet ({len(train_df)} rows)")
    print(f"      - val.parquet ({len(val_df)} rows)")
    print(f"      - test.parquet ({len(test_df)} rows)")

    # Сохраняем public/private test
    if public_test_df is not None:
        public_test_df.to_parquet(output_dir / 'public_test.parquet', index=False)
        print(f"      - public_test.parquet ({len(public_test_df)} rows)")

    if private_test_df is not None:
        private_test_df.to_parquet(output_dir / 'private_test.parquet', index=False)
        print(f"      - private_test.parquet ({len(private_test_df)} rows)")

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
        'val_end_date': str(val_end_date.date())
    }

    # Добавляем информацию о public/private test если они есть
    if public_test_df is not None:
        metadata['public_test_size'] = len(public_test_df)
        metadata['public_test_period'] = f"{public_test_df['begin'].min().date()} to {public_test_df['begin'].max().date()}"

    if private_test_df is not None:
        metadata['private_test_size'] = len(private_test_df)
        metadata['private_test_period'] = f"{private_test_df['begin'].min().date()} to {private_test_df['begin'].max().date()}"

    import json
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"   OK Saved metadata.json\n")

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

    if public_test_df is not None:
        print(f"   Public test:    {len(public_test_df)} rows")
    if private_test_df is not None:
        print(f"   Private test:   {len(private_test_df)} rows")

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

    args = parser.parse_args()

    prepare_data(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        windows=args.windows,
        include_cross_sectional=not args.no_cross_sectional,
        include_interactions=not args.no_interactions
    )
