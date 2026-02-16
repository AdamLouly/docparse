from __future__ import annotations

import json

import docparse.parser as parser_module
from docparse import DocParse, ParseConfig
from docparse.llm.base import LLMClient
from docparse.routing import choose_mode_from_page_texts


class StaticLLM(LLMClient):
    def __init__(self) -> None:
        self.calls = 0

    def complete_text(self, prompt: str, temperature: float = 0.0) -> str:
        self.calls += 1
        return json.dumps(
            {
                "page_number": 1,
                "markdown": "# Title\n\nHello world.",
                "elements": [{"type": "paragraph", "text": "Hello world."}],
                "tables": [],
            }
        )

    def complete_vision(self, prompt: str, images: list[bytes], temperature: float = 0.0) -> str:
        raise NotImplementedError


def test_auto_router_sets_chosen_mode(monkeypatch, tmp_path) -> None:
    def fake_extract_text_pages(pdf_path, max_pages=None):
        return [
            (1, "Quarterly Report\nRevenue increased by 20 percent over baseline."),
        ]

    monkeypatch.setattr(parser_module, "extract_text_pages", fake_extract_text_pages)

    llm = StaticLLM()
    config = ParseConfig(mode="auto", llm_client=llm, cache_dir=tmp_path / ".cache")
    parser = DocParse(config)

    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%docparse-test")

    document = parser.parse_pdf(pdf_path)

    assert document.metadata["requested_mode"] == "auto"
    assert document.metadata["chosen_mode"] == "text"
    assert llm.calls == 1


def test_router_prefers_ocr_for_cid_noise() -> None:
    mode, stats = choose_mode_from_page_texts(["(cid:123) (cid:456)\n12 34"])

    assert mode == "ocr"
    assert stats["has_cid_artifacts"] is True
