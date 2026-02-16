# docparse

`docparse` parses PDFs into clean Markdown and structured JSON.

## Features

- Public API: `DocParse(ParseConfig).parse_pdf(path) -> ParsedDocument`
- Modes: `text`, `ocr`, `auto`
- Output format config: `markdown` or `json`
- Provider abstraction with `LLMClient`
- OpenAI provider example implementation
- Text extraction with PyMuPDF and pdfplumber fallback
- OCR rendering with PyMuPDF page images
- Disk cache for per-page LLM results
- JSON schema validation + repair path
- Optional metadata/classification tasks with custom prompts + output schema
- Optional text chunk parsing across multiple pages per LLM call
- Streaming chunk iterator API for incremental agent pipelines
- CLI support

## Install

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e .[pdf]
pip install -e .[openai]
pip install -e .[test]
```

## Quick start (library)

```python
import os

from docparse import DocParse, MetadataTaskConfig, ParseConfig

config = ParseConfig(
    mode="auto",
    output_format="markdown",
    provider="openai",
    model="gpt-4.1-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    text_pages_per_call=1,
    metadata_tasks=[
        MetadataTaskConfig(
            name="common_core_alignment",
            prompt="Pick one or zero Common Core standards aligned with this content.",
            output_schema={
                "standard": "string",
                "confidence": "number",
            },
            target="page",
        ),
        MetadataTaskConfig(
            name="item_generation",
            prompt="Classify if this content is good for item generation.",
            output_schema={
                "is_good": "boolean",
                "reason": "string",
            },
            target="document",
        ),
    ],
)

parser = DocParse(config)
result = parser.parse_pdf("file.pdf")

print(result.markdown)
print(result.metadata)
```

Notes:

- `metadata_tasks` are optional.
- `text_pages_per_call > 1` groups sparse pages into one chunk result with `page_number` + `page_end`.
- Metadata task outputs are stored under `result.metadata["tasks"]`.

### Streaming chunks incrementally

Use `parse_pdf_chunks()` when you want to process each chunk as soon as it is ready (without waiting for the full document):

```python
from docparse import DocParse, ParseConfig

parser = DocParse(
    ParseConfig(
        mode="text",
        text_pages_per_call=4,
    )
)

for chunk in parser.parse_pdf_chunks("file.pdf"):
    print(chunk.chunk_index, chunk.page_start, chunk.page_end)
    print(chunk.markdown)
    # send chunk to downstream process immediately
```

`ParsedChunk` fields:

- `chunk_index`
- `page_start`, `page_end`
- `markdown`
- `pages` (the parsed page/chunk objects)
- `elements`
- `metadata` (includes mode info, warnings, and page-target task results)

Note: document-level metadata tasks are deferred in streaming mode and listed under `chunk.metadata["deferred_document_tasks"]`.

## CLI

```bash
python -m docparse parse file.pdf --mode auto --output markdown --provider openai --model gpt-4.1-mini
```

Parse sparse documents with chunking:

```bash
python -m docparse parse file.pdf --mode text --text-pages-per-call 5
```

Write JSON sidecar:

```bash
python -m docparse parse file.pdf --output markdown --json-sidecar
```

Explicit JSON output:

```bash
python -m docparse parse file.pdf --output json
```

## Custom LLM client

Implement `docparse.llm.LLMClient`:

- `complete_text(prompt, temperature=0) -> str`
- `complete_vision(prompt, images, temperature=0) -> str`

Then pass it via `ParseConfig(llm_client=your_client)`.
