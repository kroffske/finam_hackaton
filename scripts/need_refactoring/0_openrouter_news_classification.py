"""News classification pipeline using OpenRouter API.

This script enriches news articles with LLM-generated classifications:
- Sentiment analysis (positive/negative/neutral)
- Relevance scoring for trading decisions
- Key topic extraction
- Ticker mention verification

Usage:
    python scripts/0_openrouter_news_classification.py --sample 10
    python scripts/0_openrouter_news_classification.py --input-file train_news.csv --output-file classified_news.csv
    python scripts/0_openrouter_news_classification.py --model openai/gpt-4o --max-tokens 800
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from finam.openrouter_client import generate_with_openrouter, OpenRouterError

DEFAULT_CLASSIFICATION_PROMPT = """Ты — эксперт по анализу финансовых новостей для торговли на российском фондовом рынке.

Проанализируй новость и верни ТОЛЬКО валидный JSON (без markdown, без комментариев) со следующей структурой:
{
  "sentiment": "positive" | "negative" | "neutral",
  "relevance_score": <число от 0 до 10>,
  "key_topics": [<список ключевых тем, max 5>],
  "mentioned_tickers": [<список упомянутых тикеров: SBER, GAZP, LKOH, GMKN, NVTK, ROSN, VTBR, MTSS, MAGN, ALRS, PLZL, CHMF, MOEX, MGNT, PHOR, RUAL, AFLT, SIBN, T>],
  "reasoning": "<краткое объяснение классификации>"
}

Критерии:
- sentiment: положительная/отрицательная/нейтральная тональность для упомянутых компаний
- relevance_score: 10 = критически важно для торговли, 0 = нерелевантно
- key_topics: макроэкономика, сырье, финансы, санкции, дивиденды, слияния, регуляции и т.д.
- mentioned_tickers: только явно упомянутые компании (через название, CEO, продукты)

Отвечай СТРОГО в формате JSON."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify news articles using OpenRouter LLM API."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "participants",
        help="Directory with input news CSV files.",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="train_news.csv",
        help="Name of the input news CSV file (relative to input-dir).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "classified_news",
        help="Directory for output CSV files.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Name of output file (auto-generated from input if not specified).",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Number of news items to process (default: all). Useful for testing.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-4o-mini",
        help="OpenRouter model to use (default: gpt-4o-mini for cost efficiency).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Model temperature (default: 0.3 for more deterministic output).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=600,
        help="Maximum tokens in response (default: 600).",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom classification prompt (default: built-in financial news prompt).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay in seconds between API requests (default: 0.5 to avoid rate limits).",
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Continue processing if individual requests fail.",
    )
    return parser.parse_args(argv)


def extract_json_from_response(content: str) -> dict[str, Any] | None:
    """Extract JSON object from LLM response (handles markdown code blocks)."""
    content = content.strip()

    # Try direct JSON parse first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON from markdown code block
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find first {...} object
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def classify_news_item(
    row: pd.Series,
    model: str,
    temperature: float,
    max_tokens: int,
    prompt: str,
) -> dict[str, Any]:
    """Classify a single news item using OpenRouter API."""
    title = str(row.get("title", ""))
    text = str(row.get("publication", ""))

    try:
        result = generate_with_openrouter(
            title=title,
            text=text,
            prompt=prompt,
            model=model,
            temperature=temperature,
            extra_body={"max_tokens": max_tokens},
        )

        classification = extract_json_from_response(result["content"])

        if classification is None:
            print(f"   [WARN] Failed to parse JSON, using defaults. Raw: {result['content'][:100]}...")
            classification = {
                "sentiment": "neutral",
                "relevance_score": 0,
                "key_topics": [],
                "mentioned_tickers": [],
                "reasoning": "Failed to parse LLM response",
                "raw_response": result["content"],
            }

        classification["llm_usage_tokens"] = result["usage"].get("total_tokens", 0)
        return classification

    except OpenRouterError as exc:
        print(f"   [ERROR] OpenRouter API failed: {exc}")
        return {
            "sentiment": "neutral",
            "relevance_score": 0,
            "key_topics": [],
            "mentioned_tickers": [],
            "reasoning": f"API Error: {exc}",
            "error": str(exc),
        }


def run_classification(args: argparse.Namespace) -> None:
    """Main classification pipeline."""
    input_path = args.input_dir / args.input_file
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    print("=" * 80)
    print("NEWS CLASSIFICATION via OpenRouter")
    print("=" * 80)
    print(f"Input:  {input_path}")
    print(f"Model:  {args.model}")
    print(f"Temp:   {args.temperature}")
    print(f"Tokens: {args.max_tokens}")
    print()

    news_df = pd.read_csv(input_path)
    print(f"Loaded {len(news_df)} news items")

    if args.sample:
        news_df = news_df.head(args.sample)
        print(f"Sampling first {args.sample} items")

    prompt = args.prompt if args.prompt else DEFAULT_CLASSIFICATION_PROMPT

    classifications = []
    total = len(news_df)

    print(f"\nProcessing {total} news items...\n")

    for idx, row in news_df.iterrows():
        print(f"[{idx + 1}/{total}] Processing: {row['title'][:60]}...")

        classification = classify_news_item(
            row=row,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            prompt=prompt,
        )

        classifications.append(classification)

        print(f"   Sentiment: {classification.get('sentiment', 'N/A')}")
        print(f"   Relevance: {classification.get('relevance_score', 0)}/10")
        print(f"   Topics:    {', '.join(classification.get('key_topics', [])[:3])}")

        if not args.skip_errors and "error" in classification:
            print(f"\nERROR: Stopping due to API failure. Use --skip-errors to continue.")
            sys.exit(1)

        if idx < total - 1:
            time.sleep(args.delay)

    # Add classification columns to dataframe
    for key in ["sentiment", "relevance_score", "key_topics", "mentioned_tickers", "reasoning"]:
        news_df[f"llm_{key}"] = [c.get(key) for c in classifications]

    news_df["llm_usage_tokens"] = [c.get("llm_usage_tokens", 0) for c in classifications]

    # Save output
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.output_file:
        output_path = args.output_dir / args.output_file
    else:
        stem = Path(args.input_file).stem
        output_path = args.output_dir / f"{stem}_classified.csv"

    news_df.to_csv(output_path, index=False)

    total_tokens = news_df["llm_usage_tokens"].sum()
    avg_tokens = news_df["llm_usage_tokens"].mean()

    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Output:       {output_path}")
    print(f"Rows:         {len(news_df)}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Avg tokens:   {avg_tokens:.1f}")
    print()
    print("Sentiment distribution:")
    print(news_df["llm_sentiment"].value_counts())
    print()
    print("Relevance score statistics:")
    print(news_df["llm_relevance_score"].describe())
    print("\nDone.")


if __name__ == "__main__":
    run_classification(parse_args())
