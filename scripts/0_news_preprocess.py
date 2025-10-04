"""News preprocessing utility for the FORECAST hackathon pipeline.

Steps:
1. Load raw news CSV files.
2. Detect ticker mentions using alias lists.
3. Save enriched news datasets (optionally exploded by ticker) into preprocessed_news.
4. Report per-ticker statistics for quick diagnostics.

Usage:
    python scripts/0_news_preprocess.py --explode
    python scripts/0_news_preprocess.py --news-files train_news.csv public_test_news.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

DEFAULT_NEWS_FILES = ['train_news.csv', 'public_test_news.csv', 'private_test_news.csv', 'test_news.csv']

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finam.news_tickers import assign_news_tickers, explode_news_tickers


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess news and attach ticker matches.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "participants",
        help="Directory with raw news CSV files (default: data/raw/participants).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "preprocessed_news",
        help="Directory where preprocessed parquet files will be stored (default: data/preprocessed_news).",
    )
    parser.add_argument(
        "--news-files",
        nargs="+",
        default=DEFAULT_NEWS_FILES,
        help="List of news file names to process relative to input directory (default filters missing files).",
    )
    parser.add_argument(
        "--explode",
        action="store_true",
        help="Also save exploded (news, ticker) pairs for downstream aggregations.",
    )
    parser.add_argument(
        "--unknown-label",
        default="UNKNOWN",
        help="Label inserted when a news item has no matched ticker (default: UNKNOWN).",
    )
    parser.add_argument(
        "--suffix",
        default="_with_tickers",
        help="Suffix appended to output file stems (default: _with_tickers).",
    )
    return parser.parse_args(argv)


def process_news_file(
    input_path: Path,
    output_path: Path,
    explode: bool,
    unknown_label: str,
) -> None:
    if not input_path.exists():
        print(f"   [SKIP] File not found: {input_path}")
        return

    print(f"   [LOAD] {input_path}")
    news_df = pd.read_csv(input_path)

    enriched_df = assign_news_tickers(news_df, unknown_label=unknown_label)
    exploded_df = explode_news_tickers(enriched_df, unknown_label=unknown_label)

    has_ticker_ratio = (
        float(enriched_df["has_ticker"].mean()) if not enriched_df.empty else None
    )
    matched_count = int(enriched_df["has_ticker"].sum()) if not enriched_df.empty else 0
    ratio_display = f"{has_ticker_ratio:.1%}" if has_ticker_ratio is not None else "N/A"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    enriched_df.to_parquet(output_path, index=False)
    print(
        f"   [SAVE] {output_path} | rows={len(enriched_df)} | matched={matched_count}"
        f" ({ratio_display} with ticker)"
    )

    ticker_counts = {
        ticker: int(count)
        for ticker, count in sorted(
            exploded_df["ticker"].value_counts().items(), key=lambda x: x[1], reverse=True
        )
    }
    stats = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "rows": int(len(enriched_df)),
        "matched_rows": matched_count,
        "matched_ratio": has_ticker_ratio,
        "exploded_rows": int(len(exploded_df)),
        "ticker_counts": ticker_counts,
    }

    stats_path = output_path.with_name(output_path.stem + "_stats.json")
    with stats_path.open("w", encoding="utf-8") as fp:
        json.dump(stats, fp, ensure_ascii=False, indent=2)
    print(f"   [SAVE] {stats_path} | tracked {len(ticker_counts)} tickers")

    if ticker_counts:
        print("   [STATS] Top tickers:")
        for ticker, count in list(ticker_counts.items())[:10]:
            print(f"      {ticker}: {count}")
    else:
        print("   [STATS] No ticker matches found")

    if explode:
        exploded_path = output_path.with_name(output_path.stem + "_exploded.parquet")
        exploded_df.to_parquet(exploded_path, index=False)
        print(f"   [SAVE] {exploded_path} | rows={len(exploded_df)} (exploded)")


def run_preprocessing(args: argparse.Namespace) -> None:
    print("=" * 80)
    print("NEWS PREPROCESSING")
    print("=" * 80)
    print(f"Input directory:  {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Files: {', '.join(args.news_files)}\n")

    for filename in args.news_files:
        input_path = args.input_dir / filename
        stem = Path(filename).stem + args.suffix
        output_path = args.output_dir / f"{stem}.parquet"
        process_news_file(
            input_path=input_path,
            output_path=output_path,
            explode=args.explode,
            unknown_label=args.unknown_label,
        )

    print("\nDone.")


if __name__ == "__main__":
    run_preprocessing(parse_args())



