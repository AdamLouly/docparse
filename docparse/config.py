from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MetadataTaskConfig(BaseModel):
    name: str
    prompt: str
    output_schema: dict[str, Any] | None = None
    target: Literal["page", "document"] = "page"
    temperature: float | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("metadata task name cannot be empty")
        return normalized

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("metadata task prompt cannot be empty")
        return normalized

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if not 0.0 <= value <= 0.2:
            raise ValueError("metadata task temperature must be between 0.0 and 0.2")
        return value


class ParseConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    mode: Literal["text", "ocr", "auto"] = "auto"
    output_format: Literal["markdown", "json"] = "markdown"

    provider: str = "openai"
    model: str = "gpt-4.1-mini"
    api_key: str | None = None
    base_url: str | None = None

    temperature: float = 0.0
    max_retries: int = 3
    retry_backoff_seconds: float = 1.0

    auto_route_pages: int = 3
    ocr_dpi: int = 200
    max_pages_per_batch: int = 4
    text_pages_per_call: int = 1

    cache_dir: Path = Field(default_factory=lambda: Path(".docparse_cache"))

    llm_client: Any | None = None
    metadata_tasks: list[MetadataTaskConfig] = Field(default_factory=list)

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, value: float) -> float:
        if not 0.0 <= value <= 0.2:
            raise ValueError("temperature must be between 0.0 and 0.2")
        return value

    @field_validator("max_pages_per_batch")
    @classmethod
    def validate_batch_size(cls, value: int) -> int:
        if value < 1:
            raise ValueError("max_pages_per_batch must be >= 1")
        return value

    @field_validator("text_pages_per_call")
    @classmethod
    def validate_text_pages_per_call(cls, value: int) -> int:
        if value < 1:
            raise ValueError("text_pages_per_call must be >= 1")
        return value

    @field_validator("metadata_tasks")
    @classmethod
    def validate_unique_task_names(
        cls,
        value: list[MetadataTaskConfig],
    ) -> list[MetadataTaskConfig]:
        names = [task.name for task in value]
        if len(set(names)) != len(names):
            raise ValueError("metadata task names must be unique")
        return value
