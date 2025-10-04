"""Utility helpers for mapping news articles to tickers."""
from __future__ import annotations

import re
from typing import Mapping, Sequence

import pandas as pd

try:
    from nltk.corpus import stopwords as nltk_stopwords
    from nltk.tokenize import RegexpTokenizer
except ImportError:  # pragma: no cover - optional dependency
    nltk_stopwords = None
    RegexpTokenizer = None

DEFAULT_TICKER_NAMES: dict[str, list[str]] = {
    "SBER": [
        "сбербанк",
        "сбер",
        "sberbank",
        "sber",
        "sb ",
        "сб ",
        "сбера",
        "министерствофинансов",
        "греф",
        "сберу",
        "сбербанку",
    ],
    "GAZP": [
        "газпром",
        "gazprom",
        "газа",
        "ogzpy",
        "газпрома",
        "миллер",
        "силасибири",
        "сахалин",
        "соболево",
        "газпрому",
    ],
    "LKOH": [
        "лукойл",
        "lukoil",
        "lukoy",
        "лукойла",
        "воробьев",
        "воробьёв",
        "лукойлу",
    ],
    "GMKN": [
        "норникель",
        "норильскийникель",
        "norilsk",
        "норильск",
        "gmk",
        "горнометаллургическая",
        "потанин",
        "норникелю",
        "норильскомуникелю",
    ],
    "NVTK": [
        "новатэк",
        "novatek",
        "новатэка",
        "новатека",
        "михельсон",
        "новатэку",
        "новатеку",
    ],
    "ROSN": [
        "роснефть",
        "rosneft",
        "игорьсечин",
        "сечин",
        "роснефти",
    ],
    "VTBR": [
        "втб",
        "vtb",
        "внешторгбанк",
        "костин",
    ],
    "MTSS": [
        "мтс",
        "mts",
        "мобильныетелесистемы",
        "mobile telesystems",
        "афксистема",
        "мтсбанк",
        "юрент",
    ],
    "MAGN": [
        "магнитогорский",
        "ммк",
        "магнитка",
        "magnitogorsk",
        "рашников",
    ],
    "ALRS": [
        "алроса",
        "alrosa",
        "алмазы",
        "алмазная",
        "маринычев",
        "алросе",
    ],
    "PLZL": [
        "полюс",
        "polyus",
        "золото",
        "полюсзолото",
        "сулейман",
        "керимов",
        "полюсу",
    ],
    "CHMF": [
        "северсталь",
        "severstal",
        "мордашов",
    ],
    "MOEX": [
        "мосбиржа",
        "московскаябиржа",
        "moex",
        "биржа",
        "жидков",
    ],
    "MGNT": [
        "магнит",
        "magnit",
        "магнита",
        "ритейл",
        "галицкий",
        "магнитальянс",
        "галактиковна",
    ],
    "PHOR": [
        "фосагро",
        "phosagro",
        "удобрения",
        "гильгенберг",
    ],
    "RUAL": [
        "русал",
        "rusal",
        "алюминий",
        "дерипаска",
    ],
    "AFLT": [
        "аэрофлот",
        "aeroflot",
        "авиакомпания",
        "савельев",
    ],
    "SIBN": [
        "газпромнефть",
        "gazprom neft",
        "газпромнефть",
        "дочкагазпрома",
    ],
    "T": [
        "тинькофф",
        "tinkoff",
        "тиньков",
        "tcs",
        "тксгрупп",
        "тиньк",
        "тбанк",
        "tcs group",
        "tcs group holding plc",
        "тбанку",
        "тинькову",
        "тинькоффу",
    ],
}

if RegexpTokenizer is not None:
    _TOKENIZER = RegexpTokenizer(r"\w+", gaps=False, discard_empty=True, flags=re.UNICODE)
else:  # pragma: no cover - fallback when nltk missing
    _TOKENIZER = None

_STOPWORDS: set[str] = set()
if nltk_stopwords is not None:
    for language in ("russian", "english"):
        try:
            _STOPWORDS.update(nltk_stopwords.words(language))
        except LookupError:  # pragma: no cover - resource missing
            continue


def _tokenize(text: str) -> list[str]:
    lowered = text.lower()
    if _TOKENIZER is not None:
        tokens = _TOKENIZER.tokenize(lowered)
    else:
        tokens = re.findall(r"\w+", lowered, flags=re.UNICODE)
    if _STOPWORDS:
        tokens = [token for token in tokens if token not in _STOPWORDS]
    return tokens


def normalize_text(text: object) -> str:
    """Return a normalized representation of *text* using NLTK tokenization."""
    if text is None:
        return ""
    tokens = _tokenize(str(text))
    return " ".join(tokens)


def _build_alias_forms(
    ticker_names: Mapping[str, Sequence[str]]
) -> dict[str, tuple[set[str], set[str]]]:
    alias_forms: dict[str, tuple[set[str], set[str]]] = {}
    for ticker, aliases in ticker_names.items():
        raw_aliases: set[str] = set()
        normalized_aliases: set[str] = set()
        for alias in aliases:
            if alias is None:
                continue
            raw = str(alias).lower()
            if raw:
                raw_aliases.add(raw)
            normalized = normalize_text(alias)
            if normalized:
                normalized_aliases.add(normalized)
        alias_forms[ticker] = (raw_aliases, normalized_aliases)
    return alias_forms


_DEFAULT_ALIAS_FORMS = _build_alias_forms(DEFAULT_TICKER_NAMES)


def find_tickers_in_text(
    text: object,
    ticker_names: Mapping[str, Sequence[str]] | None = None,
) -> list[str]:
    """Return tickers whose aliases are present in *text*.

    Examples
    --------
    >>> find_tickers_in_text("Сбербанк нарастил прибыль")
    ['SBER']
    >>> find_tickers_in_text("Газпром и Лукойл подписали соглашение")
    ['GAZP', 'LKOH']
    """
    if ticker_names is None:
        ticker_names = DEFAULT_TICKER_NAMES

    alias_forms = (
        _DEFAULT_ALIAS_FORMS
        if ticker_names is DEFAULT_TICKER_NAMES
        else _build_alias_forms(ticker_names)
    )

    if text is None:
        return []

    text_lower = str(text).lower()
    normalized_text = normalize_text(text)
    normalized_haystack = f" {normalized_text} " if normalized_text else ""

    matches: list[str] = []
    for ticker, (raw_aliases, normalized_aliases) in alias_forms.items():
        raw_hit = any(alias and alias in text_lower for alias in raw_aliases)
        normalized_hit = False
        if not raw_hit and normalized_haystack:
            normalized_hit = any(
                alias and f" {alias} " in normalized_haystack for alias in normalized_aliases
            )
        if raw_hit or normalized_hit:
            if ticker not in matches:
                matches.append(ticker)
    return matches


def assign_news_tickers(
    news_df: pd.DataFrame,
    title_col: str = "title",
    text_col: str = "publication",
    ticker_names: Mapping[str, Sequence[str]] | None = None,
    unknown_label: str | None = "UNKNOWN",
) -> pd.DataFrame:
    """Annotate *news_df* with detected tickers."""
    if ticker_names is None:
        ticker_names = DEFAULT_TICKER_NAMES

    news = news_df.copy()

    missing_cols = [col for col in (title_col, text_col) if col not in news.columns]
    if missing_cols:
        raise KeyError(f"Missing columns in news_df: {missing_cols}")

    news[title_col] = news[title_col].fillna("")
    news[text_col] = news[text_col].fillna("")

    news["tickers_from_title"] = news[title_col].apply(
        lambda value: find_tickers_in_text(value, ticker_names=ticker_names)
    )
    news["tickers_from_text"] = news[text_col].apply(
        lambda value: find_tickers_in_text(value, ticker_names=ticker_names)
    )

    def _merge_row(row: pd.Series) -> list[str]:
        combined: list[str] = []
        if row["tickers_from_title"]:
            combined.extend(row["tickers_from_title"])
        if row["tickers_from_text"]:
            combined.extend(row["tickers_from_text"])
        unique_combined = list(dict.fromkeys(combined))
        if not unique_combined and unknown_label is not None:
            return [unknown_label]
        return unique_combined

    news["matched_tickers"] = news.apply(_merge_row, axis=1)
    news["has_ticker"] = news["matched_tickers"].apply(
        lambda values: bool(values) and not (len(values) == 1 and values[0] == unknown_label)
    )

    return news


def explode_news_tickers(
    news_df: pd.DataFrame,
    tickers_col: str = "matched_tickers",
    unknown_label: str | None = "UNKNOWN",
) -> pd.DataFrame:
    """Return a ticker-level view of the news DataFrame."""
    if tickers_col not in news_df.columns:
        raise KeyError(f"{tickers_col!r} column is missing. Call assign_news_tickers first.")

    exploded = news_df.explode(tickers_col).rename(columns={tickers_col: "ticker"})
    if unknown_label is not None and "ticker" in exploded.columns:
        exploded = exploded[exploded["ticker"] != unknown_label]
    return exploded
