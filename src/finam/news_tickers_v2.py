"""Simplified news ticker detection using tikers_dict.py.

Упрощенная версия без NLTK - простой поиск подстрок в тексте.
"""
from __future__ import annotations

import pandas as pd

from finam.tikers_dict import ticker_names


def find_tickers_in_text(text: str | None) -> list[str]:
    """Найти тикеры упомянутые в тексте.

    Args:
        text: Текст для поиска (title или publication)

    Returns:
        Список найденных тикеров

    Examples:
        >>> find_tickers_in_text("Сбербанк нарастил прибыль")
        ['SBER']
        >>> find_tickers_in_text("Газпром и Лукойл подписали соглашение")
        ['GAZP', 'LKOH']
    """
    if not text:
        return []

    text_lower = str(text).lower()
    found_tickers = []

    for ticker, aliases in ticker_names.items():
        for alias in aliases:
            if alias.lower() in text_lower:
                if ticker not in found_tickers:
                    found_tickers.append(ticker)
                break  # нашли тикер, переходим к следующему

    return found_tickers


def assign_tickers(
    news_df: pd.DataFrame,
    title_col: str = "title",
    text_col: str = "publication",
) -> pd.DataFrame:
    """Добавить колонку 'tickers' в DataFrame с новостями.

    Args:
        news_df: DataFrame с новостями
        title_col: Название колонки с заголовком
        text_col: Название колонки с текстом публикации

    Returns:
        DataFrame с новой колонкой 'tickers' (list[str])
    """
    news = news_df.copy()

    # Заполнить пропуски
    news[title_col] = news[title_col].fillna("")
    news[text_col] = news[text_col].fillna("")

    # Искать в заголовке и тексте
    def find_in_row(row):
        tickers_title = find_tickers_in_text(row[title_col])
        tickers_text = find_tickers_in_text(row[text_col])
        # Объединить и удалить дубликаты
        all_tickers = list(dict.fromkeys(tickers_title + tickers_text))
        return all_tickers

    news["tickers"] = news.apply(find_in_row, axis=1)

    return news
