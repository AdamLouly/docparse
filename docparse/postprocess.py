from __future__ import annotations

import re
from collections import Counter

from .models import ParsedPage

_TABLE_SEPARATOR_RE = re.compile(
    r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$"
)


def merge_pages_markdown(pages: list[ParsedPage]) -> str:
    page_markdowns = [page.markdown.strip() for page in pages if page.markdown.strip()]
    if not page_markdowns:
        return ""

    deduped = _remove_repeated_page_edges(page_markdowns)
    merged = "\n\n".join(chunk for chunk in deduped if chunk.strip())
    merged = _ensure_heading_spacing(merged)
    merged = _normalize_markdown_tables(merged)
    merged = re.sub(r"\n{3,}", "\n\n", merged)
    return merged.strip()


def _remove_repeated_page_edges(page_markdowns: list[str]) -> list[str]:
    top_counter: Counter[str] = Counter()
    bottom_counter: Counter[str] = Counter()

    split_pages: list[list[str]] = []
    for markdown in page_markdowns:
        lines = [line.rstrip() for line in markdown.splitlines() if line.strip()]
        split_pages.append(lines)
        if lines:
            top_counter[lines[0]] += 1
            bottom_counter[lines[-1]] += 1

    repeated_top = {line for line, count in top_counter.items() if count >= 2}
    repeated_bottom = {line for line, count in bottom_counter.items() if count >= 2}

    cleaned: list[str] = []
    for lines in split_pages:
        local = list(lines)
        if local and local[0] in repeated_top:
            local = local[1:]
        if local and local[-1] in repeated_bottom:
            local = local[:-1]
        cleaned.append("\n".join(local).strip())

    return cleaned


def _ensure_heading_spacing(markdown: str) -> str:
    lines = markdown.splitlines()
    output: list[str] = []

    for index, line in enumerate(lines):
        is_heading = bool(re.match(r"^\s{0,3}#{1,6}\s+\S", line))

        if is_heading and output and output[-1].strip():
            output.append("")

        output.append(line.rstrip())

        next_line = lines[index + 1] if index + 1 < len(lines) else ""
        if is_heading and next_line.strip():
            output.append("")

    return "\n".join(output)


def _normalize_markdown_tables(markdown: str) -> str:
    lines = markdown.splitlines()
    output: list[str] = []
    index = 0

    while index < len(lines):
        line = lines[index]
        if "|" not in line:
            output.append(line)
            index += 1
            continue

        start = index
        while index < len(lines) and "|" in lines[index]:
            index += 1

        block = lines[start:index]
        output.extend(_normalize_table_block(block))

    return "\n".join(output)


def _normalize_table_block(block: list[str]) -> list[str]:
    if not block:
        return block

    rows = [_split_table_row(line) for line in block]
    if not rows or not rows[0]:
        return block

    column_count = len(rows[0])

    normalized: list[list[str]] = []
    for row in rows:
        padded = list(row) + [""] * max(0, column_count - len(row))
        normalized.append(padded[:column_count])

    lines: list[str] = []
    lines.append(_format_table_row(normalized[0]))

    has_separator = len(block) > 1 and _TABLE_SEPARATOR_RE.match(block[1]) is not None
    if has_separator:
        data_rows = normalized[2:]
    else:
        data_rows = normalized[1:]

    separator = _format_table_row(["---"] * column_count)
    lines.append(separator)
    lines.extend(_format_table_row(row) for row in data_rows)
    return lines


def _split_table_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _format_table_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"
