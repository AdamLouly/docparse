from __future__ import annotations

import json
from typing import Any


def parse_llm_json(raw_output: str) -> Any:
    cleaned = raw_output.strip()
    if cleaned.startswith("```"):
        cleaned = _strip_code_fences(cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        candidate = _extract_json_candidate(cleaned)
        if candidate is None:
            raise
        return json.loads(candidate)


def _strip_code_fences(text: str) -> str:
    lines = text.strip().splitlines()
    if not lines:
        return text

    if lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_json_candidate(text: str) -> str | None:
    start = None
    opening = ""
    for idx, char in enumerate(text):
        if char == "{":
            start = idx
            opening = "{"
            break
        if char == "[":
            start = idx
            opening = "["
            break

    if start is None:
        return None

    closing = "}" if opening == "{" else "]"
    depth = 0
    in_string = False
    escaped = False

    for idx in range(start, len(text)):
        char = text[idx]

        if escaped:
            escaped = False
            continue

        if char == "\\":
            escaped = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]

    return None
