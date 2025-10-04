"""Explode news by tickers: одна строка новости → N строк (по одной на каждый тикер).

Этот скрипт:
1. Загружает *_with_tickers_llm.csv (где tickers = список)
2. Делает explode: одна строка на тикер
3. Добавляет news_type (company_specific / market_wide / market_wide_company)
4. Сохраняет *_ticker_sentiment.csv

Usage:
    # Обработать test data
    python scripts/0_3_llm_explode.py

    # Обработать train data
    python scripts/0_3_llm_explode.py --train-only

    # Обработать оба файла
    python scripts/0_3_llm_explode.py --all
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import pandas as pd

# Add src/ to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def classify_news_type(row: pd.Series) -> str:
    """
    Определить тип новости на основе тикеров.

    Args:
        row: строка DataFrame с колонками 'ticker' и 'total_tickers'

    Returns:
        news_type: 'company_specific', 'market_wide', или 'market_wide_company'

    Logic:
        - company_specific: только 1 тикер в новости
        - market_wide: тикер = 'MARKET' (общерыночная новость)
        - market_wide_company: несколько тикеров (2+) упомянуты в одной новости
    """
    if row['ticker'] == 'MARKET':
        return 'market_wide'
    elif row['total_tickers'] == 1:
        return 'company_specific'
    else:
        return 'market_wide_company'


def explode_by_tickers(news_df: pd.DataFrame) -> pd.DataFrame:
    """
    Explode новостей по тикерам: одна строка новости → N строк (по одной на тикер).

    Args:
        news_df: DataFrame с колонками [publish_date, title, publication, tickers, sentiment, confidence]
                 где tickers = "['T', 'VKCO']" (строка или список)

    Returns:
        DataFrame с колонками [publish_date, ticker, title, publication, sentiment, confidence, news_type]

    Example:
        Input:
            publish_date    | tickers            | sentiment | confidence
            2025-04-15      | ['T', 'VKCO']     | 1         | 7

        Output:
            publish_date    | ticker | sentiment | confidence | news_type
            2025-04-15      | T      | 1         | 7          | market_wide_company
            2025-04-15      | VKCO   | 1         | 7          | market_wide_company
    """
    df = news_df.copy()

    # Конвертировать tickers из строки в список если нужно
    if df['tickers'].dtype == 'object':
        df['tickers'] = df['tickers'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x
        )

    # Сохраняем количество тикеров для classification
    df['total_tickers'] = df['tickers'].apply(lambda x: len(x) if isinstance(x, list) else 0)

    # Explode: одна строка на тикер
    exploded = df.explode('tickers').rename(columns={'tickers': 'ticker'})

    # Убираем пустые тикеры
    exploded = exploded[exploded['ticker'].notna()].copy()

    # Классифицируем тип новости
    exploded['news_type'] = exploded.apply(classify_news_type, axis=1)

    # Конвертируем дату
    exploded['publish_date'] = pd.to_datetime(exploded['publish_date'])

    # Выбираем нужные колонки
    result_cols = [
        'publish_date',
        'ticker',
        'title',
        'publication',
        'sentiment',
        'confidence',
        'news_type'
    ]

    # Добавляем опциональные колонки если есть
    optional_cols = ['_hash']
    for col in optional_cols:
        if col in exploded.columns:
            result_cols.append(col)

    return exploded[result_cols].reset_index(drop=True)


def process_file(
    input_file: Path,
    output_file: Path
) -> None:
    """
    Обработать один файл: загрузить, explode, сохранить.

    Args:
        input_file: путь к *_with_tickers_llm.csv
        output_file: путь для сохранения *_ticker_sentiment.csv
    """
    print(f"\n{'='*70}")
    print(f"Processing: {input_file.name}")
    print(f"{'='*70}")

    # Проверка существования файла
    if not input_file.exists():
        print(f"[ERROR] File not found: {input_file}")
        return

    # Загрузка
    print("   Loading...")
    news_df = pd.read_csv(input_file)
    print(f"   Loaded {len(news_df):,} news items")

    # Проверка наличия необходимых колонок
    required_cols = ['publish_date', 'tickers']
    missing_cols = [col for col in required_cols if col not in news_df.columns]
    if missing_cols:
        print(f"   [ERROR] Missing columns: {missing_cols}")
        print(f"   Required columns: {required_cols}")
        return

    # Проверяем наличие LLM колонок
    has_llm = 'sentiment' in news_df.columns and 'confidence' in news_df.columns

    if has_llm:
        # Статистика по LLM
        valid_sentiment = news_df['sentiment'].notna()
        print(f"   LLM sentiment: {valid_sentiment.sum():,} / {len(news_df):,} ({valid_sentiment.sum()/len(news_df)*100:.1f}%)")
    else:
        # Добавляем пустые LLM колонки
        print("   [INFO] No LLM columns - adding empty sentiment/confidence")
        news_df['sentiment'] = pd.NA
        news_df['confidence'] = pd.NA

    # Explode по тикерам
    print("\n   Exploding by tickers...")
    exploded_df = explode_by_tickers(news_df)

    # Статистика
    print("\n   [OK] Exploded:")
    print(f"      Input rows:  {len(news_df):,}")
    print(f"      Output rows: {len(exploded_df):,}")
    print(f"      Unique tickers: {exploded_df['ticker'].nunique()}")
    print(f"      Date range: {exploded_df['publish_date'].min()} to {exploded_df['publish_date'].max()}")

    # Распределение по типам новостей
    print("\n   News type distribution:")
    for news_type, count in exploded_df['news_type'].value_counts().items():
        pct = count / len(exploded_df) * 100
        print(f"      {news_type:25s}: {count:6,} ({pct:5.1f}%)")

    # Сохранение
    output_file.parent.mkdir(parents=True, exist_ok=True)
    exploded_df.to_csv(output_file, index=False)
    print(f"\n   [SAVE] Saved to: {output_file}")

    # Пример данных
    print("\n   Sample (first 3 rows):")
    sample_cols = ['publish_date', 'ticker', 'sentiment', 'confidence', 'news_type']
    print(exploded_df[sample_cols].head(3).to_string(index=False))


def main():
    """Main pipeline."""
    parser = argparse.ArgumentParser(description='Explode news by tickers')
    parser.add_argument('--train-only', action='store_true', help='Process only train data')
    parser.add_argument('--test-only', action='store_true', help='Process only test data')
    parser.add_argument('--all', action='store_true', help='Process both train and test data')
    args = parser.parse_args()

    print("="*70)
    print("NEWS TICKER EXPLODE (LLM SENTIMENT)")
    print("="*70)

    preprocessed_dir = PROJECT_ROOT / 'data' / 'preprocessed_news'

    # Определяем какие файлы обрабатывать
    process_train = args.train_only or args.all
    process_test = args.test_only or args.all or (not args.train_only and not args.test_only)

    # ========================================================================
    # 1. Train data
    # ========================================================================
    if process_train:
        # Попробуем сначала с LLM, потом без
        train_input_llm = preprocessed_dir / 'news_with_tickers_llm.csv'
        train_input_basic = preprocessed_dir / 'news_with_tickers.csv'
        train_input = train_input_llm if train_input_llm.exists() else train_input_basic
        train_output = preprocessed_dir / 'news_ticker_sentiment.csv'
        process_file(train_input, train_output)

    # ========================================================================
    # 2. Test data
    # ========================================================================
    if process_test:
        # Попробуем сначала с LLM, потом без
        test_input_llm = preprocessed_dir / 'news_2_with_tickers_llm.csv'
        test_input_basic = preprocessed_dir / 'news_2_with_tickers.csv'
        test_input = test_input_llm if test_input_llm.exists() else test_input_basic
        test_output = preprocessed_dir / 'news_2_ticker_sentiment.csv'
        process_file(test_input, test_output)

    # ========================================================================
    # Summary
    # ========================================================================
    print(f"\n{'='*70}")
    print("EXPLODE COMPLETE!")
    print(f"{'='*70}")

    print("\nGenerated files:")
    if process_train and (preprocessed_dir / 'news_ticker_sentiment.csv').exists():
        print("   [OK] news_ticker_sentiment.csv")
    if process_test and (preprocessed_dir / 'news_2_ticker_sentiment.csv').exists():
        print("   [OK] news_2_ticker_sentiment.csv")

    print("\nNext steps:")
    print("   # Generate ticker-level features (aggregated)")
    print("   python scripts/0_1_news_ticker_features.py")


if __name__ == "__main__":
    main()
