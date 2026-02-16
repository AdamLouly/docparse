from __future__ import annotations

import os

from docparse import DocParse, MetadataTaskConfig, ParseConfig


def main() -> None:
    config = ParseConfig(
        mode="auto",
        output_format="markdown",
        provider="openai",
        model="gpt-5.2",
        api_key=os.getenv("OPENAI_API_KEY"),
        text_pages_per_call=2,
        metadata_tasks=[
            MetadataTaskConfig(
                name="item_generation",
                prompt="Tag each page with the dominant language.",
                output_schema={
                    "language": "string",
                    "reason": "string",
                },
                target="page",
            ),
        ],
    )

    parser = DocParse(config)
    for chunk in parser.parse_pdf_chunks("sample.pdf"):
        # immediately available
        print(chunk.markdown)
        print(chunk.metadata)

if __name__ == "__main__":
    main()
