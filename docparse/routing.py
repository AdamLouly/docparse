from __future__ import annotations

import re
from typing import Any


def compute_extraction_stats(page_texts: list[str]) -> dict[str, Any]:
    all_text = "\n".join(page_texts)
    total_chars = len(all_text)

    if total_chars == 0:
        return {
            "alpha_density": 0.0,
            "non_printable_ratio": 1.0,
            "cid_count": 0,
            "has_cid_artifacts": False,
            "avg_line_length": 0.0,
        }

    alpha_chars = sum(1 for char in all_text if char.isalpha())
    non_printable_chars = sum(
        1 for char in all_text if (ord(char) < 32 and char not in {"\n", "\t", "\r"})
    )
    cid_count = len(re.findall(r"\(cid:\d+\)", all_text, flags=re.IGNORECASE))

    lines = [line.strip() for line in all_text.splitlines() if line.strip()]
    if lines:
        avg_line_length = sum(len(line) for line in lines) / len(lines)
    else:
        avg_line_length = 0.0

    return {
        "alpha_density": alpha_chars / total_chars,
        "non_printable_ratio": non_printable_chars / total_chars,
        "cid_count": cid_count,
        "has_cid_artifacts": cid_count > 0,
        "avg_line_length": avg_line_length,
    }


def choose_mode_from_page_texts(page_texts: list[str]) -> tuple[str, dict[str, Any]]:
    stats = compute_extraction_stats(page_texts)

    choose_ocr = (
        stats["alpha_density"] < 0.55
        or stats["non_printable_ratio"] > 0.03
        or stats["has_cid_artifacts"]
        or stats["avg_line_length"] < 20
    )

    return ("ocr" if choose_ocr else "text"), stats
