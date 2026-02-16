from __future__ import annotations

import json

import docparse.parser as parser_module
from docparse import DocParse, ParseConfig
from docparse.llm.base import LLMClient


class RepairPathLLM(LLMClient):
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete_text(self, prompt: str, temperature: float = 0.0) -> str:
        self.prompts.append(prompt)
        if len(self.prompts) == 1:
            return "this is not valid json"

        return json.dumps(
            {
                "page_number": 1,
                "markdown": "Repaired markdown",
                "elements": [{"type": "paragraph", "text": "Repaired markdown"}],
                "tables": [],
            }
        )

    def complete_vision(self, prompt: str, images: list[bytes], temperature: float = 0.0) -> str:
        raise NotImplementedError


def test_text_schema_repair_path(monkeypatch, tmp_path) -> None:
    def fake_extract_text_pages(pdf_path, max_pages=None):
        return [(1, "A page requiring schema repair.")]

    monkeypatch.setattr(parser_module, "extract_text_pages", fake_extract_text_pages)

    llm = RepairPathLLM()
    config = ParseConfig(mode="text", llm_client=llm, cache_dir=tmp_path / ".cache")

    pdf_path = tmp_path / "repair-test.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%repair-test")

    parser = DocParse(config)
    document = parser.parse_pdf(pdf_path)

    assert len(llm.prompts) == 2
    assert "invalid for the required schema" in llm.prompts[1].lower()
    assert document.pages[0].markdown == "Repaired markdown"
