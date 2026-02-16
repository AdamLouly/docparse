from __future__ import annotations

import math
import re
from collections import Counter

_PAGE_NUMBER_RE = re.compile(r"^\s*(?:page\s+)?\d+(?:\s*(?:/|of)\s*\d+)?\s*$", re.IGNORECASE)
_CID_RE = re.compile(r"\(cid:\d+\)")


def clean_text_pages(pages: list[tuple[int, str]]) -> list[tuple[int, str]]:
    raw_texts = [text for _, text in pages]
    headers, footers = detect_repeated_headers_footers(raw_texts)

    cleaned: list[tuple[int, str]] = []
    for page_number, text in pages:
        cleaned_text = clean_page_text(text, headers=headers, footers=footers)
        cleaned.append((page_number, cleaned_text))
    return cleaned


def detect_repeated_headers_footers(page_texts: list[str]) -> tuple[set[str], set[str]]:
    if not page_texts:
        return set(), set()

    top_counter: Counter[str] = Counter()
    bottom_counter: Counter[str] = Counter()

    for page_text in page_texts:
        lines = [line.strip() for line in page_text.splitlines() if line.strip()]
        if not lines:
            continue
        top_counter[lines[0]] += 1
        bottom_counter[lines[-1]] += 1

    min_frequency = max(2, math.ceil(len(page_texts) * 0.5))
    headers = {line for line, count in top_counter.items() if count >= min_frequency}
    footers = {line for line, count in bottom_counter.items() if count >= min_frequency}
    return headers, footers


def clean_page_text(text: str, headers: set[str], footers: set[str]) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = _CID_RE.sub("", normalized)
    normalized = re.sub(r"(\w)-\n(\w)", r"\1\2", normalized)

    lines = [line.strip() for line in normalized.split("\n")]

    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()

    if lines and lines[0] in headers:
        lines = lines[1:]
    if lines and lines[-1] in footers:
        lines = lines[:-1]

    filtered: list[str] = []
    for line in lines:
        if _PAGE_NUMBER_RE.match(line):
            continue

        compact = re.sub(r"[ \t]{2,}", " ", line)
        filtered.append(compact)

    joined = "\n".join(filtered)
    joined = re.sub(r"[ \t]+\n", "\n", joined)
    joined = re.sub(r"\n{3,}", "\n\n", joined)
    joined = re.sub(r"\u0000", "", joined)
    return joined.strip()
