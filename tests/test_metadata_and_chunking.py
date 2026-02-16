from __future__ import annotations

import json

import docparse.parser as parser_module
import pytest
from docparse import DocParse, MetadataTaskConfig, ParseConfig
from docparse.llm.base import LLMClient


class ChunkingLLM(LLMClient):
    def __init__(self) -> None:
        self.calls = 0

    def complete_text(self, prompt: str, temperature: float = 0.0) -> str:
        self.calls += 1
        if '"page_start": <int>' in prompt:
            return json.dumps(
                {
                    "page_start": 1,
                    "page_end": 2,
                    "markdown": "# Combined\n\nChunked output",
                    "elements": [{"type": "paragraph", "text": "Chunked output"}],
                    "tables": [],
                }
            )

        if '"page_number": <int>' in prompt:
            return json.dumps(
                {
                    "page_number": 3,
                    "markdown": "Chunk 3",
                    "elements": [{"type": "paragraph", "text": "chunk 3"}],
                    "tables": [],
                }
            )

        raise AssertionError(f"Unexpected prompt for chunking test: {prompt[:120]}")

    def complete_vision(self, prompt: str, images: list[bytes], temperature: float = 0.0) -> str:
        raise NotImplementedError


class StreamingLLM(LLMClient):
    def __init__(self) -> None:
        self.calls = 0

    def complete_text(self, prompt: str, temperature: float = 0.0) -> str:
        self.calls += 1

        if '"page_start": <int>' in prompt:
            if "Expected page_start: 1" in prompt:
                return json.dumps(
                    {
                        "page_start": 1,
                        "page_end": 2,
                        "markdown": "Chunk 1-2",
                        "elements": [{"type": "paragraph", "text": "chunk 1"}],
                        "tables": [],
                    }
                )

            return json.dumps(
                {
                    "page_start": 3,
                    "page_end": 3,
                    "markdown": "Chunk 3",
                    "elements": [{"type": "paragraph", "text": "chunk 3"}],
                    "tables": [],
                }
            )

        if '"page_number": <int>' in prompt:
            return json.dumps(
                {
                    "page_number": 3,
                    "markdown": "Chunk 3",
                    "elements": [{"type": "paragraph", "text": "chunk 3"}],
                    "tables": [],
                }
            )

        if "Task name: chunk_tag" in prompt:
            return json.dumps({"label": "math"})

        if "Task name: doc_label" in prompt:
            return json.dumps({"kind": "worksheet"})

        raise AssertionError(f"Unexpected prompt for streaming test: {prompt[:120]}")

    def complete_vision(self, prompt: str, images: list[bytes], temperature: float = 0.0) -> str:
        raise NotImplementedError


class MetadataLLM(LLMClient):
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete_text(self, prompt: str, temperature: float = 0.0) -> str:
        self.prompts.append(prompt)

        if "The output for metadata task 'alignment' is invalid." in prompt:
            return json.dumps(
                {
                    "standard": "CCSS.2.NBT.A.1",
                    "confidence": 0.86,
                }
            )

        if "Task name: alignment" in prompt:
            return json.dumps(
                {
                    "standard": "CCSS.2.NBT.A.1",
                }
            )

        if "Task name: item_generation" in prompt:
            return json.dumps(
                {
                    "is_good": True,
                    "reason": "Content has clear constraints and numeric context.",
                }
            )

        if '"page_number": <int>' in prompt:
            return json.dumps(
                {
                    "page_number": 1,
                    "markdown": "# Parsed\n\nPage body",
                    "elements": [{"type": "paragraph", "text": "Page body"}],
                    "tables": [],
                }
            )

        raise AssertionError(f"Unexpected prompt for metadata test: {prompt[:120]}")

    def complete_vision(self, prompt: str, images: list[bytes], temperature: float = 0.0) -> str:
        raise NotImplementedError


def test_text_chunk_parsing_sets_page_range(monkeypatch, tmp_path) -> None:
    def fake_extract_text_pages(pdf_path, max_pages=None):
        return [
            (1, "First sparse page."),
            (2, "Second sparse page."),
        ]

    monkeypatch.setattr(parser_module, "extract_text_pages", fake_extract_text_pages)

    llm = ChunkingLLM()
    config = ParseConfig(
        mode="text",
        text_pages_per_call=2,
        llm_client=llm,
        cache_dir=tmp_path / ".cache",
    )

    pdf_path = tmp_path / "chunk-test.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%chunk-test")

    parser = DocParse(config)
    document = parser.parse_pdf(pdf_path)

    assert len(document.pages) == 1
    assert document.pages[0].page_number == 1
    assert document.pages[0].page_end == 2
    assert document.pages[0].markdown.startswith("# Combined")
    assert llm.calls == 1


def test_metadata_tasks_with_schema_and_repair(monkeypatch, tmp_path) -> None:
    def fake_extract_text_pages(pdf_path, max_pages=None):
        return [(1, "This page is about place value and number representations.")]

    monkeypatch.setattr(parser_module, "extract_text_pages", fake_extract_text_pages)

    llm = MetadataLLM()
    config = ParseConfig(
        mode="text",
        llm_client=llm,
        cache_dir=tmp_path / ".cache",
        metadata_tasks=[
            MetadataTaskConfig(
                name="alignment",
                prompt="Pick one or zero Common Core standards aligned to this page.",
                output_schema={
                    "standard": "string",
                    "confidence": "number",
                },
                target="page",
            ),
            MetadataTaskConfig(
                name="item_generation",
                prompt="Classify whether this page is good for item generation.",
                output_schema={
                    "is_good": "boolean",
                    "reason": "string",
                },
                target="document",
            ),
        ],
    )

    pdf_path = tmp_path / "metadata-test.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%metadata-test")

    parser = DocParse(config)
    document = parser.parse_pdf(pdf_path)

    tasks = document.metadata.get("tasks", {})
    assert tasks["alignment"]["target"] == "page"
    assert tasks["alignment"]["results"][0]["output"]["standard"] == "CCSS.2.NBT.A.1"
    assert tasks["alignment"]["results"][0]["output"]["confidence"] == 0.86

    assert tasks["item_generation"]["target"] == "document"
    assert tasks["item_generation"]["result"]["is_good"] is True

    repair_prompts = [p for p in llm.prompts if "metadata task 'alignment' is invalid" in p.lower()]
    assert len(repair_prompts) == 1


def test_parse_pdf_chunks_streams_chunk_objects(monkeypatch, tmp_path) -> None:
    def fake_extract_text_pages(pdf_path, max_pages=None):
        return [
            (1, "Page one"),
            (2, "Page two"),
            (3, "Page three"),
        ]

    monkeypatch.setattr(parser_module, "extract_text_pages", fake_extract_text_pages)

    llm = StreamingLLM()
    config = ParseConfig(
        mode="text",
        text_pages_per_call=2,
        llm_client=llm,
        cache_dir=tmp_path / ".cache",
        metadata_tasks=[
            MetadataTaskConfig(
                name="chunk_tag",
                prompt="Tag this chunk.",
                output_schema={"label": "string"},
                target="page",
            ),
            MetadataTaskConfig(
                name="doc_label",
                prompt="Label this whole document.",
                output_schema={"kind": "string"},
                target="document",
            ),
        ],
    )

    pdf_path = tmp_path / "stream-test.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%stream-test")

    parser = DocParse(config)
    stream = parser.parse_pdf_chunks(pdf_path)

    first_chunk = next(stream)
    assert first_chunk.chunk_index == 0
    assert first_chunk.page_start == 1
    assert first_chunk.page_end == 2
    assert first_chunk.metadata["tasks"]["chunk_tag"]["results"][0]["output"]["label"] == "math"
    assert first_chunk.metadata["deferred_document_tasks"] == ["doc_label"]

    second_chunk = next(stream)
    assert second_chunk.chunk_index == 1
    assert second_chunk.page_start == 3
    assert second_chunk.page_end == 3

    with pytest.raises(StopIteration):
        next(stream)


def test_parse_pdf_chunks_works_with_single_page_chunks(monkeypatch, tmp_path) -> None:
    def fake_extract_text_pages(pdf_path, max_pages=None):
        return [(1, "Page one")]

    monkeypatch.setattr(parser_module, "extract_text_pages", fake_extract_text_pages)

    llm = MetadataLLM()
    config = ParseConfig(
        mode="text",
        text_pages_per_call=1,
        llm_client=llm,
        cache_dir=tmp_path / ".cache",
        metadata_tasks=[
            MetadataTaskConfig(
                name="alignment",
                prompt="Pick one or zero Common Core standards aligned to this page.",
                output_schema={
                    "standard": "string",
                    "confidence": "number",
                },
                target="page",
            ),
        ],
    )

    pdf_path = tmp_path / "stream-single-test.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%stream-single-test")

    parser = DocParse(config)
    chunks = list(parser.parse_pdf_chunks(pdf_path))

    assert len(chunks) == 1
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 1
    assert chunks[0].metadata["tasks"]["alignment"]["results"][0]["output"]["confidence"] == 0.86
