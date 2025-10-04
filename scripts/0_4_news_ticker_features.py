"""Generate ticker-level news features from exploded ticker-sentiment data.

Этот скрипт:
1. Загружает *_ticker_sentiment.csv (уже exploded по тикерам)
2. Агрегирует на уровне (date, ticker):
   - llm_news_count_1d + дополнительные окна (3/7/14/30/60)
   - llm_sentiment_mean, llm_sentiment_weighted, llm_confidence_mean
   - llm_positive_count, llm_negative_count, llm_neutral_count
   - llm_news_type_count__*, llm_impact_scope_count__*, и доли по категориям
   - rolling features для всех метрик
3. Сохраняет *_ticker_features.csv

Usage:
    # Генерация ticker features для всех файлов
    python scripts/0_1_news_ticker_features.py

    # Только для train данных
    python scripts/0_1_news_ticker_features.py --train-only

    # Только для test данных
    python scripts/0_1_news_ticker_features.py --test-only
"""

import sys
from pathlib import Path
import argparse

# Добавляем src/ в PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

import pandas as pd  # noqa: E402

from finam.features_news_tickers import aggregate_ticker_news_features  # noqa: E402


def process_ticker_sentiment_file(
    input_file: Path,
    output_file: Path
) -> None:
    """
    Агрегировать ticker sentiment в ticker features.

    Args:
        input_file: путь к *_ticker_sentiment.csv (exploded)
        output_file: путь для сохранения *_ticker_features.csv
    """
    print(f"\n{'='*70}")
    print(f"Processing: {input_file.name}")
    print(f"{'='*70}")

    # Проверяем существование входного файла
    if not input_file.exists():
        print(f"[ERROR] File not found: {input_file}")
        print("   [INFO] Run: python scripts/0_3_llm_explode.py")
        return

    # Загружаем данные
    print("   Loading exploded ticker-sentiment data...")
    ticker_sentiment_df = pd.read_csv(input_file)
    print(f"   Loaded {len(ticker_sentiment_df):,} rows (exploded)")

    # Проверяем наличие необходимых колонок
    required_cols = ['publish_date', 'ticker']
    missing_cols = [col for col in required_cols if col not in ticker_sentiment_df.columns]
    if missing_cols:
        print(f"   [ERROR] Missing columns: {missing_cols}")
        return

    # Проверяем наличие LLM колонок
    has_llm = 'sentiment' in ticker_sentiment_df.columns and 'confidence' in ticker_sentiment_df.columns

    if has_llm:
        valid_sentiment = ticker_sentiment_df['sentiment'].notna()
        print(f"   [OK] LLM sentiment available: {valid_sentiment.sum():,} / {len(ticker_sentiment_df):,} ({valid_sentiment.sum()/len(ticker_sentiment_df)*100:.1f}%)")
    else:
        print("   [INFO] No LLM sentiment columns found")

    # Статистика
    print(f"   Unique tickers: {ticker_sentiment_df['ticker'].nunique()}")
    print(f"   Date range: {pd.to_datetime(ticker_sentiment_df['publish_date']).min()} to {pd.to_datetime(ticker_sentiment_df['publish_date']).max()}")

    # Агрегация ticker features
    print("\n   Aggregating to (date, ticker) level...")
    print(f"   - Include LLM features: {has_llm}")

    ticker_features = aggregate_ticker_news_features(
        ticker_sentiment_df,
        rolling_windows=[3, 7, 14, 30, 60],
        include_llm_features=has_llm
    )

    # Статистика
    print("\n   [OK] Generated features:")
    print(f"      Total rows: {len(ticker_features):,}")
    print(f"      Unique tickers: {ticker_features['ticker'].nunique()}")
    print(f"      Date range: {ticker_features['date'].min()} to {ticker_features['date'].max()}")
    print(f"      Features: {len(ticker_features.columns)}")

    # Список feature колонок
    feature_cols = [col for col in ticker_features.columns if col not in ['date', 'ticker']]
    print(f"\n   Feature columns ({len(feature_cols)}):")
    for i, col in enumerate(feature_cols, 1):
        print(f"      {i:2d}. {col}")

    # Сохранение
    output_file.parent.mkdir(parents=True, exist_ok=True)
    ticker_features.to_csv(output_file, index=False)
    print(f"\n   [SAVE] Saved to: {output_file}")

    # Пример данных
    print("\n   Sample (first 3 rows):")
    sample_cols = [col for col in ticker_features.columns[:8]]  # First 8 columns
    print(ticker_features[sample_cols].head(3).to_string(index=False))


def main():
    """Main pipeline."""
    parser = argparse.ArgumentParser(description='Generate ticker-level news features')
    parser.add_argument('--train-only', action='store_true', help='Process only train data')
    parser.add_argument('--test-only', action='store_true', help='Process only test data')
    args = parser.parse_args()

    print("="*70)
    print("TICKER NEWS FEATURES AGGREGATION")
    print("="*70)

    preprocessed_dir = project_root / 'data' / 'preprocessed_news'

    # Определяем какие файлы обрабатывать
    process_train = not args.test_only
    process_test = not args.train_only

    # ========================================================================
    # 1. Train data
    # ========================================================================
    if process_train:
        train_input = preprocessed_dir / 'news_ticker_sentiment.csv'
        train_output = preprocessed_dir / 'news_ticker_features.csv'

        process_ticker_sentiment_file(
            input_file=train_input,
            output_file=train_output
        )

    # ========================================================================
    # 2. Test data
    # ========================================================================
    if process_test:
        test_input = preprocessed_dir / 'news_2_ticker_sentiment.csv'
        test_output = preprocessed_dir / 'news_2_ticker_features.csv'

        process_ticker_sentiment_file(
            input_file=test_input,
            output_file=test_output
        )

    # ========================================================================
    # Summary
    # ========================================================================
    print(f"\n{'='*70}")
    print("AGGREGATION COMPLETE!")
    print(f"{'='*70}")

    print("\nGenerated files:")
    if process_train and (preprocessed_dir / 'news_ticker_features.csv').exists():
        print("   [OK] news_ticker_features.csv")
    if process_test and (preprocessed_dir / 'news_2_ticker_features.csv').exists():
        print("   [OK] news_2_ticker_features.csv")

    print("\nNext steps:")
    print("   # Run data preparation pipeline")
    print("   python scripts/1_prepare_data.py")
    print("\n   # The ticker features will be automatically loaded")
    print("   # and joined to candles data")


if __name__ == "__main__":
    main()
