"""Fast parallel LLM sentiment analysis for financial news.

Processes news articles with:
- Parallel API calls (6 concurrent requests by default)
- Automatic retry with exponential backoff
- Rate limit handling (429 responses)
- Resume support (continue from interrupted runs)
- Deduplication by content hash
- Progress tracking and cost estimation

Usage:
    # Process both train and test data (default)
    python scripts/0_2_llm_models.py

    # Process only train data
    python scripts/0_2_llm_models.py --file train

    # Process only test data
    python scripts/0_2_llm_models.py --file test

    # Filter by date (for testing)
    python scripts/0_2_llm_models.py --start-date 2025-01-01

Configuration:
    - Set OPENROUTER_API_KEY environment variable
    - Adjust CONCURRENCY, ITEMS_PER_CALL below if needed
"""

from __future__ import annotations

import argparse
import ast
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Add src/ to path for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finam.llm_sentiment import (  # noqa: E402
    AdaptiveLimiter,
    analyze_news_batch,
    hash_news_row,
    supports_structured_outputs,
)

# ============================================
# CONFIGURATION
# ============================================

# Model settings
MODEL = "google/gemini-2.5-flash-lite"  # Fast and cost-effective
PROMPT_VERSION = "v2"  # Use enhanced prompt with impact scope
MAX_TOKENS = 1000  # Sufficient for JSON responses
TEMPERATURE = 0.3  # Deterministic outputs

# Batch and concurrency settings
CONCURRENCY = 6  # Number of parallel requests (start with 4-6)
ITEMS_PER_CALL = 20  # News items per API call (50-100 recommended)
TEXT_CHARS = 400  # Max chars from publication text

# Progress and safety
SAVE_EVERY = 20  # Save checkpoint every N batches

# Data paths - will be set by command line args
FILES_CONFIG = {
    "train": {
        "input": "data/preprocessed_news/news_with_tickers.csv",
        "output": "news_with_tickers_llm",
    },
    "test": {
        "input": "data/preprocessed_news/news_2_with_tickers.csv",
        "output": "news_2_with_tickers_llm",
    },
}

# Model pricing (per 1M tokens)
MODEL_PRICES = {
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "deepseek/deepseek-r1-distill-llama-70b": {"input": 0.03, "output": 0.13},
    "qwen/qwen-2.5-72b-instruct": {"input": 0.35, "output": 0.40},
    "mistralai/mistral-small": {"input": 0.04, "output": 0.40},
    "google/gemini-2.5-flash-lite": {"input": 0.037, "output": 0.150},
}

IMPACT_SCOPE_DEFAULTS = {
    "market_wide": "market",
    "market_wide_company": "market_and_company",
    "company_specific": "company",
}


def parse_ticker_list(raw: object) -> list[str]:
    """Convert serialized tickers value to a clean list of tickers."""

    if isinstance(raw, list):
        items = raw
    elif raw is None or (isinstance(raw, float) and pd.isna(raw)):
        items = []
    elif isinstance(raw, str):
        value = raw.strip()
        if not value:
            items = []
        else:
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, list):
                    items = parsed
                else:
                    items = [str(parsed)]
            except (SyntaxError, ValueError):
                items = [tok.strip() for tok in value.split(",") if tok.strip()]
    else:
        items = [str(raw)]

    clean = []
    for item in items:
        if item is None:
            continue
        token = str(item).strip()
        if token:
            clean.append(token)
    return clean


def infer_news_type(tickers: list[str]) -> str:
    """Derive news_type from list of tickers."""

    unique = [t for t in tickers if t]
    if not unique or set(unique) == {"MARKET"}:
        return "market_wide"
    dedup = set(unique)
    if len(dedup) == 1:
        return "company_specific"
    return "market_wide_company"


def choose_primary_ticker(tickers: list[str], news_type: str) -> str:
    """Pick primary ticker for prompt context."""

    if news_type == "market_wide":
        return "MARKET"
    for ticker in tickers:
        if ticker and ticker != "MARKET":
            return ticker
    return tickers[0] if tickers else "MARKET"


def fallback_impact_scope(news_type: str) -> str:
    """Return default impact scope derived from news_type."""

    return IMPACT_SCOPE_DEFAULTS.get(news_type, "company")


def calculate_cost(
    input_tokens: int, output_tokens: int, model: str = MODEL
) -> tuple[float, float, float]:
    """Calculate API cost for given token usage.

    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model: Model identifier

    Returns:
        Tuple of (input_cost, output_cost, total_cost) in USD
    """
    prices = MODEL_PRICES.get(model, {"input": 0.15, "output": 0.60})
    input_cost = (input_tokens / 1_000_000) * prices["input"]
    output_cost = (output_tokens / 1_000_000) * prices["output"]
    return input_cost, output_cost, input_cost + output_cost


def setup_logging() -> Path:
    """Setup logging to file and console.

    Returns:
        Path to log file
    """
    # Create logs directory
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)

    # Log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"llm_sentiment_{timestamp}.log"

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    return log_file


def batch_worker(
    batch_df: pd.DataFrame, settings: dict
) -> dict[str, list | dict | None]:
    """Process single batch (called by ThreadPoolExecutor).

    Args:
        batch_df: DataFrame slice to process
        settings: Dict with 'model', 'max_tokens', 'temperature', 'text_chars'

    Returns:
        dict with 'sentiments' (list) and 'usage' (dict)
    """
    result = analyze_news_batch(
        batch_df,
        model=settings["model"],
        max_tokens=settings["max_tokens"],
        temperature=settings["temperature"],
        text_chars=settings["text_chars"],
        prompt_version=settings["prompt_version"],
    )
    return result


def process_file(
    input_file: str,
    output_prefix: str,
    start_date: str | None,
    end_date: str | None,
    logger: logging.Logger,
    force: bool = False,
) -> None:
    """Process single news file.

    Args:
        input_file: Path to input CSV file
        output_prefix: Prefix for output filename
        start_date: Optional start date filter
        end_date: Optional end date filter
        logger: Logger instance
    """
    input_path = PROJECT_ROOT / input_file
    if not input_path.exists():
        print(f"❌ File not found: {input_path}")
        return

    print(f"\n{'=' * 70}")
    print("🚀 FAST PARALLEL LLM SENTIMENT ANALYSIS")
    print(f"{'=' * 70}")

    # Check if model supports structured outputs
    use_structured = supports_structured_outputs(MODEL)

    logger.info("Starting LLM sentiment analysis")
    logger.info(f"Input file: {input_path}")
    logger.info(f"Model: {MODEL}")
    logger.info(f"Prompt version: {PROMPT_VERSION}")
    logger.info(
        f"Structured outputs: {'enabled' if use_structured else 'disabled (fallback to text parsing)'}"
    )
    logger.info(f"Concurrency: {CONCURRENCY}, Batch size: {ITEMS_PER_CALL}")

    news_df = pd.read_csv(input_path)
    news_df["publish_date"] = pd.to_datetime(news_df["publish_date"])
    original_count = len(news_df)

    if "tickers" not in news_df.columns:
        print("❌ Column 'tickers' not found in input file — cannot proceed")
        logger.error("Input file missing 'tickers' column, aborting")
        return

    # Normalize tickers representation and derive news_type / primary_ticker
    news_df["tickers"] = news_df["tickers"].apply(parse_ticker_list)

    if "news_type" in news_df.columns:
        existing_types = news_df["news_type"].fillna("").astype(str)
    else:
        existing_types = pd.Series([""] * len(news_df), index=news_df.index)

    derived_types = []
    for tickers, current_type in zip(news_df["tickers"], existing_types):
        if current_type in IMPACT_SCOPE_DEFAULTS:
            derived_types.append(current_type)
        else:
            derived_types.append(infer_news_type(tickers))
    news_df["news_type"] = derived_types

    news_df["primary_ticker"] = [
        choose_primary_ticker(tickers, ntype)
        for tickers, ntype in zip(news_df["tickers"], news_df["news_type"])
    ]

    logger.info(f"Loaded {original_count} news items")

    # Filter by date
    if start_date or end_date:
        if start_date:
            news_df = news_df[news_df["publish_date"] >= pd.to_datetime(start_date)]
        if end_date:
            end_dt = (
                pd.to_datetime(end_date)
                + pd.Timedelta(days=1)
                - pd.Timedelta(seconds=1)
            )
            news_df = news_df[news_df["publish_date"] <= end_dt]

        date_range = f"{start_date or 'start'} - {end_date or 'end'}"
    else:
        date_range = "all dates"

    news_df = news_df.reset_index(drop=True)

    if news_df.empty:
        print("❌ No news after date filtering!")
        return

    print(f"Model: {MODEL}")
    print(f"Prompt version: {PROMPT_VERSION}")
    if use_structured:
        print("✓ Structured outputs: ENABLED (JSON Schema validation)")
    else:
        print("⚠ Structured outputs: disabled (text parsing fallback)")
    print(f"Concurrency: {CONCURRENCY} parallel requests")
    print(f"Batch size: {ITEMS_PER_CALL} items/request")
    print(f"Date range: {date_range}")
    print(f"Filtered: {len(news_df):,} / {original_count:,} news")
    print(f"Est. requests: {(len(news_df) - 1) // ITEMS_PER_CALL + 1}")

    # Show pricing
    prices = MODEL_PRICES.get(MODEL, {})
    if prices:
        print("\n💰 Pricing:")
        print(f"   Input:  ${prices['input']:.3f} / 1M tokens")
        print(f"   Output: ${prices['output']:.3f} / 1M tokens")
    print(f"{'=' * 70}\n")

    # Initialize columns
    if "sentiment" not in news_df.columns:
        news_df["sentiment"] = None
    if "confidence" not in news_df.columns:
        news_df["confidence"] = None
    if "impact_scope" not in news_df.columns:
        news_df["impact_scope"] = None

    # Generate hash for deduplication
    # Hash = MD5 of (tickers + title + publication[:TEXT_CHARS])
    # Used to:
    # 1. Detect duplicate news (same content → same hash)
    # 2. Resume processing (load previous results by hash)
    news_df["_hash"] = [
        hash_news_row(row, text_chars=TEXT_CHARS)
        for row in news_df.itertuples(index=False)
    ]
    logger.info(f"Generated content hashes for {len(news_df)} items")

    # Resume from existing file if available
    output_dir = PROJECT_ROOT / "data" / "preprocessed_news"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{output_prefix}.csv"

    if output_file.exists() and not force:
        print(
            f"📂 Found existing output: {output_file.name} — skipping (use --force-llm to regenerate)"
        )
        return

    if output_file.exists():
        print(f"📂 Found existing output: {output_file.name}")
        prev_df = pd.read_csv(output_file)
        restore_cols = [
            col for col in ("sentiment", "confidence", "impact_scope") if col in prev_df.columns
        ]
        if "_hash" in prev_df.columns and restore_cols:
            prev_map: dict[str, dict[str, object]] = {}
            for row in prev_df.itertuples(index=False):
                row_hash = getattr(row, "_hash", None)
                if not isinstance(row_hash, str):
                    continue
                prev_map[row_hash] = {
                    col: getattr(row, col, None)
                    for col in restore_cols
                }

            for i, h in enumerate(news_df["_hash"]):
                if h not in prev_map:
                    continue
                cached = prev_map[h]
                if pd.notna(cached.get("sentiment")):
                    for col in restore_cols:
                        news_df.at[i, col] = cached.get(col)

            completed = news_df["sentiment"].notna().sum()
            print(f"   Resuming from {completed:,} completed items\n")

    # Find items to process
    mask = news_df["sentiment"].isna()
    todo_idx = news_df[mask].index.tolist()

    if not todo_idx:
        print("✓ All items already processed!")
        return

    # Split into batches
    batches = []
    for i in range(0, len(todo_idx), ITEMS_PER_CALL):
        batches.append(todo_idx[i : i + ITEMS_PER_CALL])

    print(f"Processing {len(todo_idx):,} remaining items in {len(batches)} batches\n")

    # Settings for workers
    settings = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "text_chars": TEXT_CHARS,
        "prompt_version": PROMPT_VERSION,
    }

    # Tracking
    total_input_tokens = 0
    total_output_tokens = 0
    limiter = AdaptiveLimiter(start=CONCURRENCY, min_c=1, max_c=CONCURRENCY * 2)
    start_time = time.time()

    # Process in parallel
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = {}
        for bnum, idxs in enumerate(batches, 1):
            batch_df = news_df.loc[
                idxs,
                [
                    "tickers",
                    "title",
                    "publication",
                    "news_type",
                    "primary_ticker",
                ],
            ].copy()
            fut = executor.submit(batch_worker, batch_df, settings)
            futures[fut] = (bnum, idxs)

        completed = 0
        for fut in as_completed(futures):
            bnum, idxs = futures[fut]
            try:
                result = fut.result()
                sentiments = result["sentiments"]
                usage = result.get("usage")

                # Update dataframe
                for local_i, global_i in enumerate(idxs):
                    response_item = sentiments[local_i] or {}

                    sentiment = response_item.get("sentiment")
                    confidence = response_item.get("confidence")
                    impact_scope = response_item.get("impact_scope")
                    news_type_llm = response_item.get("news_type")

                    news_df.at[global_i, "sentiment"] = sentiment
                    news_df.at[global_i, "confidence"] = confidence

                    if impact_scope is None:
                        impact_scope = fallback_impact_scope(
                            news_df.at[global_i, "news_type"]
                        )
                    news_df.at[global_i, "impact_scope"] = impact_scope

                    if (
                        news_type_llm in IMPACT_SCOPE_DEFAULTS
                        and news_type_llm != news_df.at[global_i, "news_type"]
                    ):
                        news_df.at[global_i, "news_type"] = news_type_llm

                # Track tokens
                if usage:
                    inp = getattr(usage, "prompt_tokens", 0)
                    out = getattr(usage, "completion_tokens", 0)
                    total_input_tokens += inp
                    total_output_tokens += out

                    _, _, cost = calculate_cost(total_input_tokens, total_output_tokens)
                    print(
                        f"✓ Batch {bnum:4d}/{len(batches)} | "
                        f"Tokens: {inp:5d}↑/{out:4d}↓ | "
                        f"Cost: ${cost:.4f}"
                    )

                limiter.record(ratelimited=False)
                completed += 1

                # Periodic save
                if completed % SAVE_EVERY == 0:
                    news_df.to_csv(output_file, index=False)
                    print(
                        f"   💾 Checkpoint saved ({completed}/{len(batches)} batches)\n"
                    )

            except Exception as e:
                # Check if rate limited
                is_rate_limit = "429" in str(e) or "rate" in str(e).lower()
                limiter.record(ratelimited=is_rate_limit)
                logger.error(f"Batch {bnum} failed: {e}", exc_info=True)
                print(f"❌ Batch {bnum} error: {e}")

    # Final save
    news_df.to_csv(output_file, index=False)
    elapsed = time.time() - start_time

    logger.info(f"Processing complete in {elapsed / 60:.1f} minutes")
    logger.info(f"Total tokens: {total_input_tokens + total_output_tokens:,}")

    # ============================================
    # RESULTS
    # ============================================
    print(f"\n{'=' * 70}")
    print("📊 RESULTS")
    print(f"{'=' * 70}")
    print(f"Model: {MODEL}")
    print(f"Date range: {date_range}")
    print(f"Total items: {len(news_df):,}")
    print(f"Processed: {news_df['sentiment'].notna().sum():,}")
    print(f"Failed: {news_df['sentiment'].isna().sum():,}")
    print(f"Time: {elapsed / 60:.1f} min")

    print("\n📈 Token usage:")
    print(f"   Input:  {total_input_tokens:,}")
    print(f"   Output: {total_output_tokens:,}")
    print(f"   Total:  {total_input_tokens + total_output_tokens:,}")

    # Cost
    if total_input_tokens > 0:
        in_cost, out_cost, total_cost = calculate_cost(
            total_input_tokens, total_output_tokens
        )
        print(f"\n💰 Cost ({MODEL}):")
        print(f"   Input:  ${in_cost:.6f}")
        print(f"   Output: ${out_cost:.6f}")
        print(f"   Total:  ${total_cost:.6f}")

        # Projection for full dataset
        if original_count > len(news_df):
            scale = original_count / len(news_df)
            proj_in = total_input_tokens * scale
            proj_out = total_output_tokens * scale
            _, _, proj_cost = calculate_cost(int(proj_in), int(proj_out))
            proj_time = elapsed * scale
            print(f"\n🔮 Full dataset projection ({original_count:,} items):")
            print(f"   Tokens: {proj_in:,.0f} input / {proj_out:,.0f} output")
            print(f"   Cost: ${proj_cost:.2f}")
            print(f"   Time: ~{proj_time / 60:.1f} min")

    # Distribution
    valid = news_df[news_df["sentiment"].notna()]
    if not valid.empty:
        print("\n📋 Sentiment distribution:")
        for sent, count in valid["sentiment"].value_counts().sort_index().items():
            label = {-1: "📉 Negative", 0: "➖ Neutral", 1: "📈 Positive"}.get(
                sent, "❓"
            )
            pct = count / len(valid) * 100
            print(f"   {label}: {count:5d} ({pct:5.1f}%)")

        avg_conf = valid["confidence"].mean()
        print(f"\n📊 Average confidence: {avg_conf:.2f}/10")

        if "impact_scope" in valid.columns:
            print("\n🌐 Impact scope distribution:")
            for scope, count in valid["impact_scope"].value_counts().items():
                pct = count / len(valid) * 100
                print(f"   {scope:18s}: {count:5d} ({pct:5.1f}%)")

        # Examples
        print("\n📰 Sample results:")
        for idx, row in valid.head(3).iterrows():
            sent_label = {1: "📈 POS", -1: "📉 NEG", 0: "➖ NEU"}.get(
                row["sentiment"], "❓"
            )
            scope_label = row.get("impact_scope", "?")
            print(
                f"\n   {sent_label} (conf: {row['confidence']}/10, scope: {scope_label}) | {row['tickers']}"
            )
            print(f"   {row['title'][:70]}...")

    print(f"\n💾 Saved to: {output_file}")
    print(f"{'=' * 70}\n")

    logger.info(f"Results saved to: {output_file}")
    logger.info("=" * 70)
    logger.info("Analysis complete")


def main() -> None:
    """Main entry point."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="LLM sentiment analysis for financial news"
    )
    parser.add_argument(
        "--file",
        choices=["train", "test", "both"],
        default="both",
        help="Which file(s) to process (default: both)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None, #"2025-01-01",
        help="Start date filter (YYYY-MM-DD) or None for all dates",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date filter (YYYY-MM-DD) or None for all dates",
    )
    parser.add_argument(
        "--force-llm",
        action="store_true",
        help="Regenerate sentiment even if cached files exist",
    )
    args = parser.parse_args()

    # Convert "None" string to None
    start_date = None if args.start_date == "None" else args.start_date
    end_date = None if args.end_date == "None" else args.end_date

    # Set console encoding for Windows
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    # Setup logging
    log_file = setup_logging()
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to: {log_file}")

    # Check API key
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        logger.error("OPENROUTER_API_KEY not found in environment!")
        print("ERROR: OPENROUTER_API_KEY not found in environment!")
        print("   Set it in .env file or export OPENROUTER_API_KEY='your-key-here'")
        sys.exit(1)

    # Determine which files to process
    if args.file == "both":
        files_to_process = ["train", "test"]
    else:
        files_to_process = [args.file]

    # Process each file
    for file_type in files_to_process:
        config = FILES_CONFIG[file_type]
        print(f"\n{'=' * 70}")
        print(f"Processing {file_type.upper()} data: {config['input']}")
        print(f"{'=' * 70}")

        process_file(
            input_file=config["input"],
            output_prefix=config["output"],
            start_date=start_date,
            end_date=end_date,
            logger=logger,
            force=args.force_llm,
        )

    print("\n\n✅ All files processed!")
    print(f"📋 Full logs: {log_file}")


if __name__ == "__main__":
    main()
