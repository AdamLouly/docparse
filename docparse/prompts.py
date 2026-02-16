from __future__ import annotations

import json
import textwrap
from typing import Any


def build_text_page_prompt(page_number: int, page_text: str) -> str:
    return textwrap.dedent(
        f"""
        You are a PDF-to-structured-markdown parser.

        Convert the page text into clean markdown and structured JSON.
        Return ONLY valid JSON (no code fences).

        Schema:
        {{
          "page_number": <int>,
          "markdown": <string>,
          "elements": [{{"type": <string>, "text": <string>}}],
          "tables": [{{"title": <string>, "rows": [[<string>, ...], ...]}}]
        }}

        Rules:
        - Keep semantic structure (headings, paragraphs, list items, tables).
        - Preserve values faithfully.
        - Put each table as rows of strings.
        - page_number MUST be {page_number}.

        Source page text:
        ---
        {page_text}
        ---
        """
    ).strip()


def build_text_chunk_prompt(page_start: int, page_end: int, chunk_text: str) -> str:
    return textwrap.dedent(
        f"""
        You are a PDF-to-structured-markdown parser.

        Convert the page chunk into clean markdown and structured JSON.
        Return ONLY valid JSON (no code fences).

        Schema:
        {{
          "page_start": <int>,
          "page_end": <int>,
          "markdown": <string>,
          "elements": [{{"type": <string>, "text": <string>}}],
          "tables": [{{"title": <string>, "rows": [[<string>, ...], ...]}}]
        }}

        Rules:
        - This chunk includes pages {page_start} through {page_end}.
        - Keep semantic structure (headings, paragraphs, list items, tables).
        - Preserve values faithfully.
        - Put each table as rows of strings.
        - page_start MUST be {page_start}.
        - page_end MUST be {page_end}.

        Source page chunk text:
        ---
        {chunk_text}
        ---
        """
    ).strip()


def build_text_repair_prompt(raw_output: str, expected_page_number: int) -> str:
    return textwrap.dedent(
        f"""
        The following output is invalid for the required schema.
        Repair it and return ONLY valid JSON (no code fences).

        Required schema:
        {{
          "page_number": <int>,
          "markdown": <string>,
          "elements": [{{"type": <string>, "text": <string>}}],
          "tables": [{{"title": <string>, "rows": [[<string>, ...], ...]}}]
        }}

        Expected page number: {expected_page_number}

        Invalid output:
        ---
        {raw_output}
        ---
        """
    ).strip()


def build_text_chunk_repair_prompt(
    raw_output: str,
    expected_page_start: int,
    expected_page_end: int,
) -> str:
    return textwrap.dedent(
        f"""
        The following output is invalid for the required schema.
        Repair it and return ONLY valid JSON (no code fences).

        Required schema:
        {{
          "page_start": <int>,
          "page_end": <int>,
          "markdown": <string>,
          "elements": [{{"type": <string>, "text": <string>}}],
          "tables": [{{"title": <string>, "rows": [[<string>, ...], ...]}}]
        }}

        Expected page_start: {expected_page_start}
        Expected page_end: {expected_page_end}

        Invalid output:
        ---
        {raw_output}
        ---
        """
    ).strip()


def build_ocr_batch_prompt(page_numbers: list[int]) -> str:
    numbers = ", ".join(str(page_number) for page_number in page_numbers)
    return textwrap.dedent(
        f"""
        You are a vision parser for PDF pages.

        You are given images for pages in this exact order: {numbers}.

        Extract each page into clean markdown and structured data.
        Return ONLY a JSON array where each item uses this schema:

        {{
          "page_number": <int>,
          "markdown": <string>,
          "elements": [{{"type": <string>, "text": <string>}}],
          "tables": [{{"title": <string>, "rows": [[<string>, ...], ...]}}]
        }}

        The array must include one object per page and page_number must match the listed pages.
        """
    ).strip()


def build_ocr_repair_prompt(raw_output: str, expected_page_numbers: list[int]) -> str:
    numbers = ", ".join(str(page_number) for page_number in expected_page_numbers)
    return textwrap.dedent(
        f"""
        The model output below is invalid.
        Repair it and return ONLY valid JSON array (no code fences).

        Expected page numbers: [{numbers}]

        Each item schema:
        {{
          "page_number": <int>,
          "markdown": <string>,
          "elements": [{{"type": <string>, "text": <string>}}],
          "tables": [{{"title": <string>, "rows": [[<string>, ...], ...]}}]
        }}

        Invalid output:
        ---
        {raw_output}
        ---
        """
    ).strip()


def build_ocr_cache_prompt(page_number: int, dpi: int) -> str:
    return f"ocr_page={page_number}|dpi={dpi}|schema=v1"


def build_page_metadata_prompt(
    task_name: str,
    task_prompt: str,
    output_schema: dict[str, Any] | None,
    page_markdown: str,
    page_start: int,
    page_end: int | None,
) -> str:
    if page_end is None or page_end == page_start:
        page_label = f"page {page_start}"
    else:
        page_label = f"pages {page_start}-{page_end}"

    schema_json = json.dumps(output_schema or _default_task_output_schema(), ensure_ascii=True, indent=2)

    return textwrap.dedent(
        f"""
        You are a document analysis assistant.

        Task name: {task_name}
        Task instruction:
        {task_prompt}

        Analyze the parsed markdown for {page_label}.

        Return ONLY valid JSON object (no code fences), following this output schema:
        {schema_json}

        Parsed markdown context:
        ---
        {page_markdown}
        ---
        """
    ).strip()


def build_document_metadata_prompt(
    task_name: str,
    task_prompt: str,
    output_schema: dict[str, Any] | None,
    document_markdown: str,
) -> str:
    schema_json = json.dumps(output_schema or _default_task_output_schema(), ensure_ascii=True, indent=2)

    return textwrap.dedent(
        f"""
        You are a document analysis assistant.

        Task name: {task_name}
        Task instruction:
        {task_prompt}

        Analyze the full parsed document markdown.

        Return ONLY valid JSON object (no code fences), following this output schema:
        {schema_json}

        Parsed markdown context:
        ---
        {document_markdown}
        ---
        """
    ).strip()


def build_metadata_repair_prompt(
    task_name: str,
    output_schema: dict[str, Any] | None,
    raw_output: str,
) -> str:
    schema_json = json.dumps(output_schema or _default_task_output_schema(), ensure_ascii=True, indent=2)

    return textwrap.dedent(
        f"""
        The output for metadata task '{task_name}' is invalid.
        Repair it and return ONLY valid JSON object (no code fences).

        Required output schema:
        {schema_json}

        Invalid output:
        ---
        {raw_output}
        ---
        """
    ).strip()


def _default_task_output_schema() -> dict[str, str]:
    return {
        "result": "string",
    }
