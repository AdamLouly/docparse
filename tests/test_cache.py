from __future__ import annotations

import json

import docparse.parser as parser_module
from docparse import DocParse, ParseConfig
from docparse.llm.base import LLMClient


class CountingLLM(LLMClient):
    def __init__(self) -> None:
        self.calls = 0

    def complete_text(self, prompt: str, temperature: float = 0.0) -> str:
        self.calls += 1
        return json.dumps(
            {
                "page_number": 1,
                "markdown": "Cached page",
                "elements": [{"type": "paragraph", "text": "Cached page"}],
                "tables": [],
            }
        )

    def complete_vision(self, prompt: str, images: list[bytes], temperature: float = 0.0) -> str:
        raise NotImplementedError


def test_disk_cache_hit_skips_llm(monkeypatch, tmp_path) -> None:
    def fake_extract_text_pages(pdf_path, max_pages=None):
        return [(1, "Header\nA stable body line for caching.")]

    monkeypatch.setattr(parser_module, "extract_text_pages", fake_extract_text_pages)

    llm = CountingLLM()
    config = ParseConfig(
        mode="text",
        llm_client=llm,
        cache_dir=tmp_path / ".docparse_cache",
    )

    pdf_path = tmp_path / "cache-test.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%cache-test")

    first = DocParse(config)
    second = DocParse(config)

    first.parse_pdf(pdf_path)
    second.parse_pdf(pdf_path)

    assert llm.calls == 1
