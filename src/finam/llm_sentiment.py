"""LLM-based sentiment analysis for financial news.

Provides parallel batch processing with retry logic, rate limit handling,
and adaptive concurrency control.

Example:
    >>> from finam.llm_sentiment import analyze_news_batch, call_llm_with_retry
    >>> batch_df = news_df.iloc[:80]
    >>> results = analyze_news_batch(batch_df, model="openai/gpt-4o-mini")
    >>> print(results['sentiments'])  # [{sentiment: -1, confidence: 8}, ...]
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import re
import time
from collections import deque
from threading import Lock
from typing import Any

import pandas as pd

# Module logger
logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The 'openai' library is required for LLM sentiment analysis. "
        "Install it with: pip install openai"
    ) from exc


# System prompt for sentiment analysis
SYSTEM_PROMPT = (
    "Ты финансовый аналитик. Для каждой строки верни sentiment ∈{-1,0,1} и confidence ∈[0..10]. "
    "Критерии:\n"
    "-1: негатив (падение, санкции, убытки, иски, негативные прогнозы)\n"
    " 0: нейтрально (факты без явного влияния)\n"
    " 1: позитив (рост прибыли, контракты, позитивные прогнозы, дивиденды)\n"
    "Выводи СТРОГО JSON-массив объектов: [{\"index\": i, \"sentiment\": s, \"confidence\": c}, ...]."
)

# Shortened prompt for structured outputs (schema enforces format)
SYSTEM_PROMPT_STRUCTURED = (
    "Ты финансовый аналитик. Для каждой новости определи:\n"
    "sentiment: -1 (негатив: падение, санкции, убытки), 0 (нейтрально), 1 (позитив: рост, контракты, дивиденды)\n"
    "confidence: 0-10 (сила влияния на акции)"
)

# JSON Schema for structured outputs
SENTIMENT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "sentiment_analysis",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "description": "Sentiment analysis results for each news item",
                    "items": {
                        "type": "object",
                        "properties": {
                            "index": {
                                "type": "integer",
                                "description": "News item index (0-based)"
                            },
                            "sentiment": {
                                "type": "integer",
                                "enum": [-1, 0, 1],
                                "description": "Sentiment: -1 negative, 0 neutral, 1 positive"
                            },
                            "confidence": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": 10,
                                "description": "Confidence level from 0 (weak) to 10 (strong)"
                            }
                        },
                        "required": ["index", "sentiment", "confidence"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["results"],
            "additionalProperties": False
        }
    }
}

# Models that support structured outputs (JSON Schema)
MODELS_WITH_STRUCTURED_OUTPUTS = {
    # OpenAI models
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "openai/gpt-4o-2024-08-06",
    "openai/gpt-4o-2024-11-20",
    "openai/chatgpt-4o-latest",
    # Add Fireworks models as needed
}


def supports_structured_outputs(model: str) -> bool:
    """Check if model supports structured outputs (JSON Schema).

    Args:
        model: Model identifier (e.g., "openai/gpt-4o-mini")

    Returns:
        True if model supports response_format with json_schema

    Example:
        >>> supports_structured_outputs("openai/gpt-4o-mini")
        True
        >>> supports_structured_outputs("anthropic/claude-3-opus")
        False
    """
    return model in MODELS_WITH_STRUCTURED_OUTPUTS


class AdaptiveLimiter:
    """Adaptive concurrency limiter that adjusts based on rate limit responses.

    Automatically reduces concurrency when rate limits are hit and gradually
    increases when requests succeed.

    Args:
        start: Initial concurrency level
        min_c: Minimum concurrency (never go below this)
        max_c: Maximum concurrency (never go above this)
        window: Number of recent requests to track
        high_ratio: If ratio of rate-limited requests exceeds this, reduce concurrency

    Example:
        >>> limiter = AdaptiveLimiter(start=6, min_c=1, max_c=12)
        >>> limiter.record(ratelimited=False)
        >>> current = limiter.get_concurrency()  # May increase if no rate limits
    """

    def __init__(
        self,
        start: int = 6,
        min_c: int = 1,
        max_c: int = 12,
        window: int = 30,
        high_ratio: float = 0.25,
    ):
        self.target = start
        self.min_c = min_c
        self.max_c = max_c
        self.window = window
        self.high_ratio = high_ratio
        self.events = deque(maxlen=window)  # True = was rate limited
        self.lock = Lock()

    def record(self, ratelimited: bool) -> None:
        """Record a request outcome (rate-limited or not)."""
        with self.lock:
            self.events.append(bool(ratelimited))
            cnt = len(self.events)
            if cnt >= max(8, self.window // 2):
                ratio = sum(self.events) / cnt
                if ratio >= self.high_ratio and self.target > self.min_c:
                    self.target = max(self.min_c, self.target - 1)
                elif ratio == 0 and self.target < self.max_c:
                    self.target = min(self.max_c, self.target + 1)

    def get_concurrency(self) -> int:
        """Get current target concurrency level."""
        with self.lock:
            return self.target


def sleep_with_jitter(base_seconds: float, attempt: int, cap: float = 60.0) -> float:
    """Sleep with exponential backoff and full jitter.

    Args:
        base_seconds: Base delay in seconds
        attempt: Current retry attempt (0-indexed)
        cap: Maximum delay in seconds

    Returns:
        Actual sleep duration
    """
    backoff = min(cap, base_seconds * (2**attempt))
    pause = random.uniform(0, backoff)  # full jitter
    time.sleep(pause)
    return pause


def parse_retry_after(exc: Exception, fallback_seconds: float) -> float:
    """Extract Retry-After header from exception or return fallback.

    Args:
        exc: Exception from HTTP client/SDK
        fallback_seconds: Default delay if header not found

    Returns:
        Delay in seconds
    """
    try:
        resp = getattr(exc, "response", None)
        if resp and getattr(resp, "headers", None):
            ra = resp.headers.get("retry-after") or resp.headers.get("Retry-After")
            if ra:
                try:
                    return float(ra)
                except Exception:
                    pass
    except Exception:
        pass
    return fallback_seconds


def compact_prompt(batch_df: pd.DataFrame, text_chars: int = 220, structured: bool = False) -> str:
    """Generate compact prompt for news batch.

    Uses CSV-like format without markdown to minimize tokens.

    Args:
        batch_df: DataFrame with columns ['tickers', 'title', 'publication']
        text_chars: Maximum characters to include from publication text
        structured: If True, omit JSON format instructions (schema handles it)

    Returns:
        Compact prompt string

    Example:
        >>> prompt = compact_prompt(batch_df)
        >>> print(prompt[:100])
        Проанализируй новости. Формат входа: index|ticker|title|text...
    """
    if structured:
        lines = ["Проанализируй новости (index|ticker|title|text):"]
    else:
        lines = [
            "Проанализируй новости. Формат входа: index|ticker|title|text (≤220 символов)."
        ]

    for local_idx, row in enumerate(batch_df.itertuples(index=False)):
        tkr = str(getattr(row, "tickers", "") or "")[:30].replace("\n", " ")
        title = str(getattr(row, "title", "") or "").replace("\n", " ").strip()
        text = (
            str(getattr(row, "publication", "") or "")
            .replace("\n", " ")
            .strip()[:text_chars]
        )
        lines.append(f"{local_idx}|{tkr}|{title}|{text}")

    if not structured:
        lines.append(f"Верни ТОЛЬКО JSON массив для index 0..{len(batch_df)-1}.")

    return "\n".join(lines)


def parse_json_array(text: str, expected_len: int, structured: bool = False) -> list[dict[str, Any]]:
    """Parse JSON array from LLM response, handling markdown blocks.

    Args:
        text: Raw LLM response text
        expected_len: Expected number of items in array
        structured: If True, expect {"results": [...]} format, else [...]

    Returns:
        List of dicts with 'index', 'sentiment', 'confidence' keys

    Example:
        >>> result = parse_json_array('```json\\n[{"index": 0, "sentiment": 1}]\\n```', 1)
        >>> print(result[0]['sentiment'])
        1
        >>> result = parse_json_array('{"results": [{"index": 0, "sentiment": 1, "confidence": 7}]}', 1, structured=True)
        >>> print(result[0]['sentiment'])
        1
    """
    original_text = text

    try:
        # For structured outputs, parse directly (no markdown, guaranteed JSON)
        if structured:
            data = json.loads(text)
            # Extract results array from {"results": [...]}
            if isinstance(data, dict) and "results" in data:
                data = data["results"]
            else:
                logger.warning(f"Structured output missing 'results' key: {text[:200]}")
                # Fallback: maybe it's already an array
                if not isinstance(data, list):
                    raise ValueError(f"Expected results array, got: {type(data)}")
        else:
            # Legacy mode: remove markdown, extract array
            if "```" in text:
                parts = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text)
                if parts:
                    text = parts[0].strip()

            # Extract array
            m = re.search(r"\[[\s\S]*\]", text)
            if m:
                text = m.group(0)

            data = json.loads(text)

    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"JSON parse error: {e}")
        logger.error(f"Raw response (first 500 chars): {original_text[:500]}")
        logger.error(f"Extracted text: {text[:200] if not structured else 'N/A'}")
        raise

    # Ensure correct length
    if len(data) != expected_len:
        out = []
        for i in range(expected_len):
            if i < len(data) and isinstance(data[i], dict):
                out.append(
                    {
                        "index": i,
                        "sentiment": data[i].get("sentiment"),
                        "confidence": data[i].get("confidence"),
                    }
                )
            else:
                out.append({"index": i, "sentiment": None, "confidence": None})
        return out
    return data


def call_llm_with_retry(
    client: OpenAI,
    payload: dict[str, Any],
    max_retries: int = 7,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    use_structured_outputs: bool = False,
) -> dict[str, Any]:
    """Call LLM with retry logic for rate limits and server errors.

    Handles:
    - 429 Rate Limit: Respects Retry-After header or uses exponential backoff
    - 5xx Server Errors: Exponential backoff with jitter
    - Network errors: Exponential backoff

    Args:
        client: OpenAI client instance
        payload: Request payload for chat.completions.create()
        max_retries: Maximum number of retry attempts
        base_delay: Base delay for exponential backoff (seconds)
        max_delay: Maximum delay between retries (seconds)
        use_structured_outputs: If True, add response_format for JSON Schema

    Returns:
        dict with 'text' (response content) and 'usage' (token stats)

    Raises:
        Exception: If all retries exhausted

    Example:
        >>> client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
        >>> payload = {"model": "openai/gpt-4o-mini", "messages": [...]}
        >>> result = call_llm_with_retry(client, payload, use_structured_outputs=True)
        >>> print(result['text'])
    """
    # Add structured outputs schema if requested
    if use_structured_outputs:
        payload = payload.copy()  # Don't modify original
        payload["response_format"] = SENTIMENT_SCHEMA
        logger.info("Using structured outputs (JSON Schema)")
    # Try to import specific exception types
    try:
        from openai import APIStatusError, RateLimitError
    except ImportError:
        RateLimitError = Exception
        APIStatusError = Exception

    last_err = None

    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(**payload)
            txt = resp.choices[0].message.content
            usage = resp.usage
            return {"text": txt, "usage": usage}

        except RateLimitError as e:
            # 429: Read Retry-After or use backoff
            ra = parse_retry_after(e, fallback_seconds=min(max_delay, base_delay * (2**attempt)))
            time_slept = min(max_delay, ra)
            # Add jitter to avoid thundering herd
            time_slept = sleep_with_jitter(time_slept, 0, max_delay)
            logger.warning(f"Rate limit hit (429), sleeping {time_slept:.1f}s, attempt {attempt+1}/{max_retries+1}")
            last_err = e

        except APIStatusError as e:
            # 5xx: backoff
            status = getattr(e, "status_code", None)
            if status and 500 <= int(status) < 600:
                sleep_with_jitter(base_delay, attempt, max_delay)
                logger.warning(f"Server error {status}, retrying attempt {attempt+1}/{max_retries+1}")
                last_err = e
            else:
                # 4xx (except 429): don't retry
                logger.error(f"Client error {status}: {e}")
                raise

        except Exception as e:
            # Network/temporary errors: backoff
            logger.warning(f"Request error: {e}, retrying attempt {attempt+1}/{max_retries+1}")
            last_err = e
            sleep_with_jitter(base_delay, attempt, max_delay)

    # All retries exhausted
    raise last_err


def hash_news_row(row: pd.Series, text_chars: int = 220) -> str:
    """Generate hash for news row to detect duplicates.

    Args:
        row: DataFrame row with 'tickers', 'title', 'publication'
        text_chars: Number of chars from publication to include in hash

    Returns:
        MD5 hash hex string

    Example:
        >>> row = pd.Series({'tickers': 'SBER', 'title': 'News', 'publication': 'Text'})
        >>> hash_val = hash_news_row(row)
        >>> print(len(hash_val))
        32
    """
    h = hashlib.md5()
    key = f"{row.tickers}|{row.title}|{str(row.publication)[:text_chars]}".encode(
        "utf-8", "ignore"
    )
    h.update(key)
    return h.hexdigest()


def analyze_news_batch(
    batch_df: pd.DataFrame,
    *,
    api_key: str | None = None,
    model: str = "openai/gpt-4o-mini",
    max_tokens: int = 1000,
    temperature: float = 0,
    text_chars: int = 220,
    force_structured: bool | None = None,
) -> dict[str, Any]:
    """Analyze sentiment for a batch of news articles.

    Args:
        batch_df: DataFrame with columns ['tickers', 'title', 'publication']
        api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
        model: Model identifier
        max_tokens: Maximum response tokens
        temperature: Sampling temperature
        text_chars: Max chars from publication text
        force_structured: Force structured outputs on/off (None=auto-detect)

    Returns:
        dict with 'sentiments' (list of dicts) and 'usage' (token stats)

    Example:
        >>> batch = news_df.iloc[:80]
        >>> result = analyze_news_batch(batch, model="openai/gpt-4o-mini")
        >>> print(result['sentiments'][0])
        {'index': 0, 'sentiment': 1, 'confidence': 7}
    """
    if api_key is None:
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY not found in environment. "
                "Set it or pass api_key parameter."
            )

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    # Determine if we should use structured outputs
    use_structured = (
        force_structured
        if force_structured is not None
        else supports_structured_outputs(model)
    )

    # Select appropriate prompt and generate user message
    system_prompt = SYSTEM_PROMPT_STRUCTURED if use_structured else SYSTEM_PROMPT
    user_prompt = compact_prompt(batch_df, text_chars=text_chars, structured=use_structured)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    logger.info(
        f"Processing batch of {len(batch_df)} items "
        f"(structured={'yes' if use_structured else 'no'})"
    )

    result = call_llm_with_retry(client, payload, use_structured_outputs=use_structured)
    sentiments = parse_json_array(
        result["text"], expected_len=len(batch_df), structured=use_structured
    )

    return {"sentiments": sentiments, "usage": result.get("usage")}
