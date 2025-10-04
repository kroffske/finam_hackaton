"""Analyze ticker alias coverage across news datasets."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finam.news_tickers import (
    assign_news_tickers,
    explode_news_tickers,
    normalize_text,
)

TEXT_COLUMNS_DEFAULT = ("title", "publication")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect token frequencies to refine ticker alias dictionaries."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "participants",
        help="Directory containing raw or preprocessed news files.",
    )
    parser.add_argument(
        "--news-files",
        nargs="+",
        default=["train_news.csv", "test_news.csv"],
        help="News file names relative to the input directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "analysis" / "news_ticker_analysis.json",
        help="Path where JSON summary will be stored.",
    )
    parser.add_argument(
        "--unknown-label",
        default="UNKNOWN",
        help="Label used for rows without ticker matches (default: UNKNOWN).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of tokens to display per category (default: 20).",
    )
    parser.add_argument(
        "--text-cols",
        nargs="+",
        default=list(TEXT_COLUMNS_DEFAULT),
        help="Columns combined to compute token frequencies (default: title publication).",
    )
    return parser.parse_args(argv)


def load_news(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file extension: {path.suffix}")


def combine_text_columns(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    available = [col for col in columns if col in df.columns]
    if not available:
        raise KeyError(f"None of the requested text columns are present: {columns}")
    return df[available].fillna("").agg(" ".join, axis=1)


def counter_from_tokens(token_lists: Iterable[list[str]]) -> Counter:
    counter: Counter[str] = Counter()
    for tokens in token_lists:
        if tokens:
            counter.update(tokens)
    return counter


def series_to_token_lists(text_series: pd.Series) -> pd.Series:
    return text_series.apply(lambda value: normalize_text(value).split() if value else [])


def analyze_file(
    path: Path,
    text_cols: list[str],
    unknown_label: str,
    top_n: int,
) -> dict:
    df = load_news(path)

    if "matched_tickers" not in df.columns:
        df = assign_news_tickers(df, unknown_label=unknown_label)

    text_series = combine_text_columns(df, text_cols)
    token_lists = series_to_token_lists(text_series)

    df = df.copy()
    df["token_list"] = token_lists

    if "has_ticker" in df.columns:
        has_ticker_series = df["has_ticker"].astype(bool)
    else:
        has_ticker_series = pd.Series(False, index=df.index)

    all_tokens = counter_from_tokens(df["token_list"])

    unknown_mask = ~has_ticker_series
    unknown_tokens = counter_from_tokens(df.loc[unknown_mask, "token_list"])

    exploded = explode_news_tickers(df, unknown_label=unknown_label)
    exploded = exploded.join(df["token_list"], how="left")

    per_ticker: dict[str, list[tuple[str, int]]] = {}
    if not exploded.empty:
        for ticker, group in exploded.groupby("ticker"):
            per_ticker[ticker] = counter_from_tokens(group["token_list"]).most_common(top_n)

    summary = {
        "file": str(path),
        "rows": int(len(df)),
        "matched_rows": int(has_ticker_series.sum()),
        "matched_ratio": float(has_ticker_series.mean()) if len(df) else 0.0,
        "top_tokens_all": all_tokens.most_common(top_n),
        "top_tokens_unknown": unknown_tokens.most_common(top_n),
        "top_tokens_by_ticker": per_ticker,
    }

    print(f"[REPORT] {path}")
    print(
        f"   rows={summary['rows']} | matched={summary['matched_rows']}"
        f" ({summary['matched_ratio']:.1%} with ticker)"
    )
    if summary["top_tokens_unknown"]:
        print("   Unknown ticker candidates:")
        for token, count in summary["top_tokens_unknown"][: min(10, top_n)]:
            print(f"      {token}: {count}")
    else:
        print("   No unknown ticker candidates detected.")

    return summary


def run_analysis(args: argparse.Namespace) -> None:
    summaries = []
    for filename in args.news_files:
        path = args.input_dir / filename
        try:
            summary = analyze_file(
                path=path,
                text_cols=args.text_cols,
                unknown_label=args.unknown_label,
                top_n=args.top_n,
            )
            summaries.append(summary)
        except FileNotFoundError as exc:
            print(f"[SKIP] {exc}")
        except ValueError as exc:
            print(f"[SKIP] {exc}")

    if not summaries:
        print("No files processed. Nothing to write.")
        return

    output_data = {"files": summaries}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fp:
        json.dump(output_data, fp, ensure_ascii=False, indent=2)
    print(f"\nSaved analysis report to {args.output}")


if __name__ == "__main__":
    run_analysis(parse_args())
