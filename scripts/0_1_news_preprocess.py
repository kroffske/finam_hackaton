"""Упрощенный препроцессинг новостей.

Загружает news.csv и news_2.csv, добавляет колонку tickers, сохраняет в preprocessed_news/.

Usage:
    python scripts/0_news_preprocess.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finam.news_tickers_v2 import assign_tickers


def preprocess_news_file(input_path: Path, output_dir: Path) -> None:
    """Обработать один файл новостей: добавить tickers, сохранить в CSV.

    Args:
        input_path: Путь к CSV файлу с новостями
        output_dir: Папка для сохранения результата
    """
    if not input_path.exists():
        print(f"[SKIP] File not found: {input_path}")
        return

    print(f"[LOAD] {input_path}")
    news_df = pd.read_csv(input_path)

    # Добавить колонку tickers
    news_with_tickers = assign_tickers(news_df)

    # Статистика
    total_rows = len(news_with_tickers)
    rows_with_tickers = (news_with_tickers["tickers"].str.len() > 0).sum()
    ratio = rows_with_tickers / total_rows if total_rows > 0 else 0

    print(
        f"  Total rows: {total_rows} | "
        f"With tickers: {rows_with_tickers} ({ratio:.1%})"
    )

    # Сохранить
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}_with_tickers.csv"
    news_with_tickers.to_csv(output_path, index=False)
    print(f"[SAVE] {output_path}")

    # Топ-5 тикеров
    all_tickers = [t for tickers in news_with_tickers["tickers"] for t in tickers]
    if all_tickers:
        ticker_counts = pd.Series(all_tickers).value_counts().head(5)
        print("  Top 5 tickers:")
        for ticker, count in ticker_counts.items():
            print(f"    {ticker}: {count}")
    print()


def main() -> None:
    input_dir = PROJECT_ROOT / "data" / "raw" / "participants"
    output_dir = PROJECT_ROOT / "data" / "preprocessed_news"

    print("=" * 80)
    print("NEWS PREPROCESSING")
    print("=" * 80)
    print(f"Input dir:  {input_dir}")
    print(f"Output dir: {output_dir}\n")

    # Обработать оба файла
    for filename in ["news.csv", "news_2.csv"]:
        input_path = input_dir / filename
        preprocess_news_file(input_path, output_dir)

    print("Done.")


if __name__ == "__main__":
    main()
