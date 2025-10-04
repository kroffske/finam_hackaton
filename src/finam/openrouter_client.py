"""OpenRouter API client for LLM-based text generation.

This module provides a simple interface to OpenRouter's chat completions API,
compatible with OpenAI's message format.

Example:
    >>> result = generate_with_openrouter(
    ...     title="News Classification",
    ...     text="Сбербанк увеличил прибыль на 15%",
    ...     prompt="Classify the sentiment as positive, negative, or neutral.",
    ...     model="openai/gpt-4o-mini",
    ... )
    >>> print(result["content"])
"""
from __future__ import annotations

import os
from typing import Any

try:
    import requests
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The 'requests' library is required for OpenRouter client. "
        "Install it with: pip install requests"
    ) from exc


class OpenRouterError(RuntimeError):
    """Exception raised when OpenRouter API requests fail."""

    pass


def generate_with_openrouter(
    *,
    title: str,
    text: str,
    prompt: str = "Ты — полезный ассистент.",
    model: str = "openai/gpt-4o-mini",
    temperature: float = 0.7,
    http_referer: str | None = None,
    timeout: float = 60.0,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate text completion using OpenRouter API.

    Args:
        title: News title or request identifier (used in X-Title header and user message).
        text: Main text content to process.
        prompt: System instruction for the model.
        model: OpenRouter model identifier (e.g., "openai/gpt-4o-mini").
        temperature: Sampling temperature (0.0-2.0).
        http_referer: Application URL for attribution (optional).
        timeout: HTTP request timeout in seconds.
        extra_body: Additional request body fields (max_tokens, top_p, seed, etc.).

    Returns:
        dict: Response containing:
            - content (str): Generated text
            - usage (dict): Token usage statistics
            - raw (dict): Full API response

    Raises:
        OpenRouterError: If API key is missing or request fails.

    Example:
        >>> result = generate_with_openrouter(
        ...     title="Sentiment Analysis",
        ...     text="Газпром подписал контракт на поставку газа в Китай",
        ...     prompt="Classify sentiment: positive, negative, or neutral. Return JSON.",
        ...     model="openai/gpt-4o-mini",
        ...     temperature=0.3,
        ...     extra_body={"max_tokens": 500},
        ... )
        >>> print(result["content"])
        {"sentiment": "positive"}
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise OpenRouterError(
            "OPENROUTER_API_KEY environment variable not found. "
            "Set it in your .env file or export it in your shell."
        )

    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": (title[:120] if title else "App"),
    }
    if http_referer:
        headers["HTTP-Referer"] = http_referer

    user_message = f"{title}\n\n{text}" if title else text

    body: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_message},
        ],
    }
    if extra_body:
        body.update(extra_body)

    try:
        resp = requests.post(url, headers=headers, json=body, timeout=timeout)
    except requests.RequestException as exc:
        raise OpenRouterError(f"Network/HTTP error: {exc}") from exc

    if resp.status_code >= 400:
        try:
            err = resp.json()
        except Exception:
            err = resp.text
        raise OpenRouterError(f"OpenRouter returned {resp.status_code}: {err}")

    data = resp.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise OpenRouterError(f"Unexpected response structure: {data}") from exc

    return {"content": content, "usage": data.get("usage", {}), "raw": data}
