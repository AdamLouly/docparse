from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RawElement(BaseModel):
    type: str
    text: str


class Element(BaseModel):
    type: str
    text: str
    page_number: int


class Table(BaseModel):
    title: str = ""
    rows: list[list[str]] = Field(default_factory=list)

    def to_markdown(self) -> str:
        if not self.rows:
            return ""
        header = self.rows[0]
        body = self.rows[1:]
        lines: list[str] = []
        if self.title:
            lines.append(f"### {self.title}")
        lines.append("| " + " | ".join(str(c) for c in header) + " |")
        lines.append("| " + " | ".join("---" for _ in header) + " |")
        for row in body:
            normalized = list(row) + [""] * max(0, len(header) - len(row))
            lines.append("| " + " | ".join(str(c) for c in normalized[: len(header)]) + " |")
        return "\n".join(lines)


class ParsedPage(BaseModel):
    page_number: int
    page_end: int | None = None
    markdown: str
    elements: list[Element] = Field(default_factory=list)
    tables: list[Table] = Field(default_factory=list)


class ParsedDocument(BaseModel):
    markdown: str
    pages: list[ParsedPage] = Field(default_factory=list)
    elements: list[Element] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParsedChunk(BaseModel):
    chunk_index: int
    page_start: int
    page_end: int
    markdown: str
    pages: list[ParsedPage] = Field(default_factory=list)
    elements: list[Element] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMPage(BaseModel):
    page_number: int
    markdown: str
    elements: list[RawElement] = Field(default_factory=list)
    tables: list[Table] = Field(default_factory=list)


class LLMPageChunk(BaseModel):
    page_start: int
    page_end: int
    markdown: str
    elements: list[RawElement] = Field(default_factory=list)
    tables: list[Table] = Field(default_factory=list)


class LLMPageBatch(BaseModel):
    pages: list[LLMPage] = Field(default_factory=list)
