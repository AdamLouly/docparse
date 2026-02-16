from __future__ import annotations

from abc import ABC, abstractmethod


class LLMClient(ABC):
    @abstractmethod
    def complete_text(self, prompt: str, temperature: float = 0.0) -> str:
        """Complete a text-only prompt and return raw model output."""

    @abstractmethod
    def complete_vision(self, prompt: str, images: list[bytes], temperature: float = 0.0) -> str:
        """Complete a multimodal prompt with image bytes and return raw model output."""
