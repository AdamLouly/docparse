from __future__ import annotations

import base64
import time
from typing import Any

from .base import LLMClient
from ..errors import ProviderError


class OpenAIClient(LLMClient):
    def __init__(
        self,
        model: str,
        api_key: str | None,
        base_url: str | None = None,
        max_retries: int = 3,
        retry_backoff_seconds: float = 1.0,
    ) -> None:
        if not api_key:
            raise ProviderError("api_key is required for provider='openai'.")

        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on optional package.
            raise ProviderError(
                "openai package is not installed. Install with: pip install openai"
            ) from exc

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds

    def complete_text(self, prompt: str, temperature: float = 0.0) -> str:
        response = self._with_retries(
            lambda: self._client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
        )
        return self._extract_response_text(response)

    def complete_vision(self, prompt: str, images: list[bytes], temperature: float = 0.0) -> str:
        message_content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image in images:
            encoded = base64.b64encode(image).decode("ascii")
            message_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded}"},
                }
            )

        response = self._with_retries(
            lambda: self._client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                messages=[{"role": "user", "content": message_content}],
            )
        )
        return self._extract_response_text(response)

    def _with_retries(self, fn: Any) -> Any:
        for attempt in range(self.max_retries):
            try:
                return fn()
            except Exception as exc:  # pragma: no cover - provider specific.
                should_retry = self._is_retryable(exc)
                is_last_attempt = attempt >= self.max_retries - 1
                if not should_retry or is_last_attempt:
                    raise ProviderError(f"OpenAI request failed: {exc}") from exc
                delay = self.retry_backoff_seconds * (2**attempt)
                time.sleep(delay)

        raise ProviderError("OpenAI request failed after retries.")

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        if status_code == 429:
            return True
        message = str(exc).lower()
        retry_markers = ["rate limit", "429", "timeout", "temporarily unavailable"]
        return any(marker in message for marker in retry_markers)

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        try:
            content = response.choices[0].message.content
        except Exception as exc:  # pragma: no cover - provider specific.
            raise ProviderError("OpenAI returned an unexpected response shape.") from exc

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and "text" in item:
                    parts.append(str(item["text"]))
                else:
                    maybe_text = getattr(item, "text", None)
                    if maybe_text:
                        parts.append(str(maybe_text))
            return "".join(parts)

        if content is None:
            return ""

        return str(content)
