from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from pydantic import ValidationError as PydanticValidationError

from .config import ParseConfig
from .errors import DocParseError
from .parser import DocParse


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="docparse", description="Parse PDF files to markdown/JSON")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_cmd = subparsers.add_parser("parse", help="Parse a PDF file")
    parse_cmd.add_argument("file", help="Path to the input PDF")
    parse_cmd.add_argument("--mode", choices=["text", "ocr", "auto"], default="auto")
    parse_cmd.add_argument(
        "--output",
        dest="output_format",
        choices=["markdown", "json"],
        default="markdown",
    )
    parse_cmd.add_argument("--provider", default="openai")
    parse_cmd.add_argument("--model", default="gpt-4.1-mini")
    parse_cmd.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parse_cmd.add_argument("--base-url", default=None)
    parse_cmd.add_argument("--temperature", type=float, default=0.0)
    parse_cmd.add_argument("--ocr-dpi", type=int, default=200)
    parse_cmd.add_argument("--max-pages-per-batch", type=int, default=4)
    parse_cmd.add_argument("--text-pages-per-call", type=int, default=1)
    parse_cmd.add_argument(
        "--json-sidecar",
        nargs="?",
        const="",
        default=None,
        help="Write parsed JSON to a sidecar file. Optionally provide output path.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_cli()
    args = parser.parse_args(argv)

    if args.command == "parse":
        return _run_parse(args)

    parser.error(f"Unsupported command: {args.command}")
    return 2


def _run_parse(args: argparse.Namespace) -> int:
    try:
        config = ParseConfig(
            mode=args.mode,
            output_format=args.output_format,
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            temperature=args.temperature,
            ocr_dpi=args.ocr_dpi,
            max_pages_per_batch=args.max_pages_per_batch,
            text_pages_per_call=args.text_pages_per_call,
        )
        parser = DocParse(config)
        document = parser.parse_pdf(args.file)
    except (DocParseError, ValueError, PydanticValidationError) as exc:
        print(f"docparse error: {exc}", file=sys.stderr)
        return 2

    if args.output_format == "json":
        print(document.model_dump_json(indent=2))
    else:
        print(document.markdown)

    if args.json_sidecar is not None:
        sidecar_path = _resolve_sidecar_path(args.file, args.json_sidecar)
        sidecar_path.write_text(document.model_dump_json(indent=2), encoding="utf-8")
        print(f"Wrote JSON sidecar: {sidecar_path}", file=sys.stderr)

    return 0


def _resolve_sidecar_path(input_pdf: str, json_sidecar_arg: str) -> Path:
    if json_sidecar_arg:
        return Path(json_sidecar_arg)

    pdf_path = Path(input_pdf)
    return pdf_path.with_suffix(".docparse.json")
