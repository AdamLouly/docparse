from __future__ import annotations

from .base import LLMClient
from .openai_client import OpenAIClient
from ..config import ParseConfig
from ..errors import ProviderError


def build_llm_client(config: ParseConfig) -> LLMClient:
    if config.llm_client is not None:
        return config.llm_client

    provider = config.provider.lower().strip()
    if provider == "openai":
        return OpenAIClient(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            max_retries=config.max_retries,
            retry_backoff_seconds=config.retry_backoff_seconds,
        )

    raise ProviderError(
        f"Unsupported provider '{config.provider}'. Only 'openai' is implemented in this package."
    )
