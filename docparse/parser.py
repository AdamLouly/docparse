from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Iterator, TypeVar

from pydantic import ValidationError as PydanticValidationError

from .cache import DiskCache
from .cleaning import clean_text_pages
from .config import MetadataTaskConfig, ParseConfig
from .errors import DocParseError, OCRUnavailableError, ProviderError, ValidationError
from .json_utils import parse_llm_json
from .llm import build_llm_client
from .models import Element, LLMPage, LLMPageChunk, ParsedChunk, ParsedDocument, ParsedPage
from .pdf import extract_text_pages, render_pages_to_png
from .postprocess import merge_pages_markdown
from .prompts import (
    build_document_metadata_prompt,
    build_metadata_repair_prompt,
    build_ocr_batch_prompt,
    build_ocr_cache_prompt,
    build_ocr_repair_prompt,
    build_page_metadata_prompt,
    build_text_chunk_prompt,
    build_text_chunk_repair_prompt,
    build_text_page_prompt,
    build_text_repair_prompt,
)
from .routing import choose_mode_from_page_texts


class DocParse:
    def __init__(self, config: ParseConfig) -> None:
        self.config = config
        self.llm_client = build_llm_client(config)
        self.cache = DiskCache(config.cache_dir)

    def parse_pdf(self, path: str | Path) -> ParsedDocument:
        pdf_path = Path(path)
        if not pdf_path.exists():
            raise DocParseError(f"PDF file does not exist: {pdf_path}")

        file_bytes = pdf_path.read_bytes()
        file_hash = self.cache.hash_bytes(file_bytes)

        requested_mode = self.config.mode
        chosen_mode, routing_stats = self._resolve_mode(pdf_path)
        warnings_list: list[str] = []

        if chosen_mode == "ocr":
            try:
                pages = self._parse_with_ocr(pdf_path, file_hash)
            except (OCRUnavailableError, NotImplementedError) as exc:
                warning_message = (
                    f"OCR/vision pipeline unavailable ({exc}). Falling back to text mode."
                )
                warnings.warn(warning_message, RuntimeWarning, stacklevel=2)
                warnings_list.append(warning_message)
                chosen_mode = "text"
                pages = self._parse_with_text(pdf_path, file_hash)
            except ProviderError as exc:
                if not self._looks_like_vision_unavailable(str(exc)):
                    raise
                warning_message = (
                    f"OCR/vision model call unavailable ({exc}). Falling back to text mode."
                )
                warnings.warn(warning_message, RuntimeWarning, stacklevel=2)
                warnings_list.append(warning_message)
                chosen_mode = "text"
                pages = self._parse_with_text(pdf_path, file_hash)
        else:
            pages = self._parse_with_text(pdf_path, file_hash)

        merged_markdown = merge_pages_markdown(pages)
        elements = [element for page in pages for element in page.elements]

        metadata: dict[str, Any] = {
            "requested_mode": requested_mode,
            "chosen_mode": chosen_mode,
            "routing": routing_stats,
            "provider": self.config.provider,
            "model": self.config.model,
            "output_format": self.config.output_format,
        }

        task_results = self._run_metadata_tasks(
            pages=pages,
            merged_markdown=merged_markdown,
            file_hash=file_hash,
            warnings_list=warnings_list,
        )
        if task_results:
            metadata["tasks"] = task_results
        if warnings_list:
            metadata["warnings"] = warnings_list

        return ParsedDocument(
            markdown=merged_markdown,
            pages=pages,
            elements=elements,
            metadata=metadata,
        )

    def parse_pdf_chunks(self, path: str | Path) -> Iterator[ParsedChunk]:
        pdf_path = Path(path)
        if not pdf_path.exists():
            raise DocParseError(f"PDF file does not exist: {pdf_path}")

        file_bytes = pdf_path.read_bytes()
        file_hash = self.cache.hash_bytes(file_bytes)

        requested_mode = self.config.mode
        chosen_mode, routing_stats = self._resolve_mode(pdf_path)
        warnings_list: list[str] = []
        chunk_index = 0

        if chosen_mode == "ocr":
            try:
                for chunk_pages in self._iter_ocr_batches(pdf_path, file_hash):
                    task_results = self._run_page_metadata_tasks_for_chunk(
                        pages=chunk_pages,
                        file_hash=file_hash,
                        warnings_list=warnings_list,
                    )
                    yield self._build_parsed_chunk(
                        chunk_index=chunk_index,
                        pages=chunk_pages,
                        requested_mode=requested_mode,
                        chosen_mode=chosen_mode,
                        routing_stats=routing_stats,
                        warnings_list=warnings_list,
                        task_results=task_results,
                    )
                    chunk_index += 1
                return
            except (OCRUnavailableError, NotImplementedError) as exc:
                warning_message = (
                    f"OCR/vision pipeline unavailable ({exc}). Falling back to text mode."
                )
                warnings.warn(warning_message, RuntimeWarning, stacklevel=2)
                warnings_list.append(warning_message)
                chosen_mode = "text"
            except ProviderError as exc:
                if not self._looks_like_vision_unavailable(str(exc)):
                    raise
                warning_message = (
                    f"OCR/vision model call unavailable ({exc}). Falling back to text mode."
                )
                warnings.warn(warning_message, RuntimeWarning, stacklevel=2)
                warnings_list.append(warning_message)
                chosen_mode = "text"

        for chunk_pages in self._iter_text_chunks(pdf_path, file_hash):
            task_results = self._run_page_metadata_tasks_for_chunk(
                pages=chunk_pages,
                file_hash=file_hash,
                warnings_list=warnings_list,
            )
            yield self._build_parsed_chunk(
                chunk_index=chunk_index,
                pages=chunk_pages,
                requested_mode=requested_mode,
                chosen_mode=chosen_mode,
                routing_stats=routing_stats,
                warnings_list=warnings_list,
                task_results=task_results,
            )
            chunk_index += 1

    def _resolve_mode(self, pdf_path: Path) -> tuple[str, dict[str, Any]]:
        if self.config.mode in {"text", "ocr"}:
            return self.config.mode, {}

        try:
            probe_pages = extract_text_pages(pdf_path, max_pages=self.config.auto_route_pages)
            probe_texts = [text for _, text in probe_pages]
            mode, stats = choose_mode_from_page_texts(probe_texts)
            return mode, stats
        except DocParseError as exc:
            return "ocr", {"router_error": str(exc)}

    def _parse_with_text(self, pdf_path: Path, file_hash: str) -> list[ParsedPage]:
        parsed_pages: list[ParsedPage] = []
        for chunk_pages in self._iter_text_chunks(pdf_path, file_hash):
            parsed_pages.extend(chunk_pages)

        return parsed_pages

    def _iter_text_chunks(self, pdf_path: Path, file_hash: str) -> Iterator[list[ParsedPage]]:
        extracted_pages = extract_text_pages(pdf_path)
        cleaned_pages = clean_text_pages(extracted_pages)
        page_chunks = _chunked(cleaned_pages, self.config.text_pages_per_call)

        for chunk in page_chunks:
            if not chunk:
                continue

            if len(chunk) == 1:
                page_number, cleaned_text = chunk[0]
                prompt = build_text_page_prompt(page_number, cleaned_text)
                cache_key = self.cache.build_key(
                    file_hash=file_hash,
                    page_number=page_number,
                    mode="text",
                    model=self.config.model,
                    prompt=prompt,
                )

                cached = self.cache.get(cache_key)
                if cached is not None:
                    llm_page = self._validate_page_payload(cached, expected_page_number=page_number)
                else:
                    raw_output = self.llm_client.complete_text(prompt, temperature=self.config.temperature)
                    llm_page = self._validate_or_repair_text_page(
                        raw_output,
                        expected_page_number=page_number,
                    )
                    self.cache.set(cache_key, llm_page.model_dump(mode="json"))

                yield [self._to_parsed_page(llm_page)]
                continue

            yield [self._parse_text_chunk(chunk, file_hash)]

    def _parse_text_chunk(self, chunk: list[tuple[int, str]], file_hash: str) -> ParsedPage:
        page_start = chunk[0][0]
        page_end = chunk[-1][0]
        chunk_text = _build_chunk_text(chunk)
        prompt = build_text_chunk_prompt(page_start, page_end, chunk_text)

        cache_key = self.cache.build_key(
            file_hash=file_hash,
            page_number=page_start,
            mode="text_chunk",
            model=self.config.model,
            prompt=prompt,
        )

        cached = self.cache.get(cache_key)
        if cached is not None:
            llm_chunk = self._validate_chunk_payload(
                cached,
                expected_page_start=page_start,
                expected_page_end=page_end,
            )
        else:
            raw_output = self.llm_client.complete_text(prompt, temperature=self.config.temperature)
            llm_chunk = self._validate_or_repair_text_chunk(
                raw_output,
                expected_page_start=page_start,
                expected_page_end=page_end,
            )
            self.cache.set(cache_key, llm_chunk.model_dump(mode="json"))

        return self._to_parsed_chunk(llm_chunk)

    def _parse_with_ocr(self, pdf_path: Path, file_hash: str) -> list[ParsedPage]:
        parsed_pages: list[ParsedPage] = []
        for batch_pages in self._iter_ocr_batches(pdf_path, file_hash):
            parsed_pages.extend(batch_pages)
        return parsed_pages

    def _iter_ocr_batches(self, pdf_path: Path, file_hash: str) -> Iterator[list[ParsedPage]]:
        rendered_pages = render_pages_to_png(pdf_path, dpi=self.config.ocr_dpi)
        if not rendered_pages:
            return

        image_by_page = {page_number: image_bytes for page_number, image_bytes in rendered_pages}
        ordered_page_numbers = [page_number for page_number, _ in rendered_pages]

        for batch_page_numbers in _chunked(ordered_page_numbers, self.config.max_pages_per_batch):
            cached_pages: dict[int, LLMPage] = {}
            uncached_page_numbers: list[int] = []

            for page_number in batch_page_numbers:
                cache_prompt = build_ocr_cache_prompt(page_number, self.config.ocr_dpi)
                cache_key = self.cache.build_key(
                    file_hash=file_hash,
                    page_number=page_number,
                    mode="ocr",
                    model=self.config.model,
                    prompt=cache_prompt,
                )
                cached = self.cache.get(cache_key)
                if cached is None:
                    uncached_page_numbers.append(page_number)
                else:
                    cached_pages[page_number] = self._validate_page_payload(
                        cached,
                        expected_page_number=page_number,
                    )

            if uncached_page_numbers:
                batch_prompt = build_ocr_batch_prompt(uncached_page_numbers)
                batch_images = [image_by_page[page_number] for page_number in uncached_page_numbers]
                raw_output = self.llm_client.complete_vision(
                    batch_prompt,
                    batch_images,
                    temperature=self.config.temperature,
                )

                validated_batch = self._validate_or_repair_ocr_batch(
                    raw_output,
                    expected_page_numbers=uncached_page_numbers,
                )

                for llm_page in validated_batch:
                    cache_prompt = build_ocr_cache_prompt(llm_page.page_number, self.config.ocr_dpi)
                    cache_key = self.cache.build_key(
                        file_hash=file_hash,
                        page_number=llm_page.page_number,
                        mode="ocr",
                        model=self.config.model,
                        prompt=cache_prompt,
                    )
                    self.cache.set(cache_key, llm_page.model_dump(mode="json"))
                    cached_pages[llm_page.page_number] = llm_page

            batch_parsed_pages: list[ParsedPage] = []
            for page_number in batch_page_numbers:
                llm_page = cached_pages.get(page_number)
                if llm_page is None:
                    raise ValidationError(
                        f"OCR batch did not return page {page_number} after validation."
                    )
                batch_parsed_pages.append(self._to_parsed_page(llm_page))

            yield batch_parsed_pages

    def _run_metadata_tasks(
        self,
        pages: list[ParsedPage],
        merged_markdown: str,
        file_hash: str,
        warnings_list: list[str],
    ) -> dict[str, Any]:
        if not self.config.metadata_tasks:
            return {}

        task_results = self._run_page_metadata_tasks_for_chunk(
            pages=pages,
            file_hash=file_hash,
            warnings_list=warnings_list,
        )

        for task in self.config.metadata_tasks:
            if task.target != "document":
                continue

            try:
                task_results[task.name] = {
                    "target": "document",
                    "result": self._run_document_metadata_task(
                        task=task,
                        merged_markdown=merged_markdown,
                        file_hash=file_hash,
                    ),
                }
            except Exception as exc:
                warning_message = f"Metadata task '{task.name}' failed: {exc}"
                warnings.warn(warning_message, RuntimeWarning, stacklevel=2)
                warnings_list.append(warning_message)

        return task_results

    def _run_page_metadata_tasks_for_chunk(
        self,
        pages: list[ParsedPage],
        file_hash: str,
        warnings_list: list[str],
    ) -> dict[str, Any]:
        if not self.config.metadata_tasks:
            return {}

        task_results: dict[str, Any] = {}
        for task in self.config.metadata_tasks:
            if task.target != "page":
                continue

            try:
                page_results = self._run_page_metadata_task(
                    task=task,
                    pages=pages,
                    file_hash=file_hash,
                )
                task_results[task.name] = {
                    "target": "page",
                    "results": page_results,
                }
            except Exception as exc:
                warning_message = f"Metadata task '{task.name}' failed: {exc}"
                warnings.warn(warning_message, RuntimeWarning, stacklevel=2)
                warnings_list.append(warning_message)

        return task_results

    def _build_parsed_chunk(
        self,
        chunk_index: int,
        pages: list[ParsedPage],
        requested_mode: str,
        chosen_mode: str,
        routing_stats: dict[str, Any],
        warnings_list: list[str],
        task_results: dict[str, Any],
    ) -> ParsedChunk:
        if not pages:
            raise ValidationError("Cannot build ParsedChunk from empty page list.")

        page_start = pages[0].page_number
        last_page = pages[-1]
        page_end = last_page.page_end if last_page.page_end is not None else last_page.page_number

        chunk_markdown = merge_pages_markdown(pages)
        chunk_elements = [element for page in pages for element in page.elements]

        metadata: dict[str, Any] = {
            "requested_mode": requested_mode,
            "chosen_mode": chosen_mode,
            "routing": routing_stats,
            "provider": self.config.provider,
            "model": self.config.model,
            "output_format": self.config.output_format,
        }
        if task_results:
            metadata["tasks"] = task_results

        deferred_document_tasks = [task.name for task in self.config.metadata_tasks if task.target == "document"]
        if deferred_document_tasks:
            metadata["deferred_document_tasks"] = deferred_document_tasks

        if warnings_list:
            metadata["warnings"] = list(warnings_list)

        return ParsedChunk(
            chunk_index=chunk_index,
            page_start=page_start,
            page_end=page_end,
            markdown=chunk_markdown,
            pages=pages,
            elements=chunk_elements,
            metadata=metadata,
        )

    def _run_page_metadata_task(
        self,
        task: MetadataTaskConfig,
        pages: list[ParsedPage],
        file_hash: str,
    ) -> list[dict[str, Any]]:
        outputs: list[dict[str, Any]] = []
        for page in pages:
            prompt = build_page_metadata_prompt(
                task_name=task.name,
                task_prompt=task.prompt,
                output_schema=task.output_schema,
                page_markdown=page.markdown,
                page_start=page.page_number,
                page_end=page.page_end,
            )
            cache_key = self.cache.build_key(
                file_hash=file_hash,
                page_number=page.page_number,
                mode=f"metadata_page:{task.name}",
                model=self.config.model,
                prompt=prompt,
            )

            cached = self.cache.get(cache_key)
            if cached is not None:
                validated_output = self._validate_metadata_payload(cached, task)
            else:
                raw_output = self.llm_client.complete_text(
                    prompt,
                    temperature=task.temperature if task.temperature is not None else self.config.temperature,
                )
                validated_output = self._validate_or_repair_metadata_output(
                    task,
                    raw_output,
                )
                self.cache.set(cache_key, validated_output)

            item: dict[str, Any] = {
                "page_number": page.page_number,
                "output": validated_output,
            }
            if page.page_end is not None and page.page_end != page.page_number:
                item["page_end"] = page.page_end
            outputs.append(item)

        return outputs

    def _run_document_metadata_task(
        self,
        task: MetadataTaskConfig,
        merged_markdown: str,
        file_hash: str,
    ) -> dict[str, Any]:
        prompt = build_document_metadata_prompt(
            task_name=task.name,
            task_prompt=task.prompt,
            output_schema=task.output_schema,
            document_markdown=merged_markdown,
        )
        cache_key = self.cache.build_key(
            file_hash=file_hash,
            page_number=0,
            mode=f"metadata_document:{task.name}",
            model=self.config.model,
            prompt=prompt,
        )

        cached = self.cache.get(cache_key)
        if cached is not None:
            return self._validate_metadata_payload(cached, task)

        raw_output = self.llm_client.complete_text(
            prompt,
            temperature=task.temperature if task.temperature is not None else self.config.temperature,
        )
        validated_output = self._validate_or_repair_metadata_output(task, raw_output)
        self.cache.set(cache_key, validated_output)
        return validated_output

    def _validate_or_repair_metadata_output(
        self,
        task: MetadataTaskConfig,
        raw_output: str,
    ) -> dict[str, Any]:
        try:
            payload = parse_llm_json(raw_output)
            return self._validate_metadata_payload(payload, task)
        except Exception as first_error:
            repair_prompt = build_metadata_repair_prompt(
                task_name=task.name,
                output_schema=task.output_schema,
                raw_output=raw_output,
            )
            repaired_output = self.llm_client.complete_text(repair_prompt, temperature=0.0)

            try:
                repaired_payload = parse_llm_json(repaired_output)
                return self._validate_metadata_payload(repaired_payload, task)
            except Exception as second_error:
                raise ValidationError(
                    f"Failed metadata task '{task.name}' validation/repair. "
                    f"Initial error: {first_error}; Repair error: {second_error}"
                ) from second_error

    def _validate_metadata_payload(
        self,
        payload: Any,
        task: MetadataTaskConfig,
    ) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValidationError(
                f"Metadata task '{task.name}' output must be a JSON object."
            )

        if task.output_schema is not None:
            self._validate_payload_against_schema(
                payload=payload,
                schema=task.output_schema,
                path=f"task:{task.name}",
            )

        return payload

    def _validate_payload_against_schema(
        self,
        payload: Any,
        schema: Any,
        path: str,
    ) -> None:
        if isinstance(schema, dict):
            if not isinstance(payload, dict):
                raise ValidationError(f"Expected object at {path}, got {type(payload).__name__}.")

            for key, child_schema in schema.items():
                if key not in payload:
                    raise ValidationError(f"Missing required key '{key}' at {path}.")
                self._validate_payload_against_schema(
                    payload=payload[key],
                    schema=child_schema,
                    path=f"{path}.{key}",
                )
            return

        if isinstance(schema, list):
            if not isinstance(payload, list):
                raise ValidationError(f"Expected array at {path}, got {type(payload).__name__}.")

            if schema:
                item_schema = schema[0]
                for index, item in enumerate(payload):
                    self._validate_payload_against_schema(
                        payload=item,
                        schema=item_schema,
                        path=f"{path}[{index}]",
                    )
            return

        if isinstance(schema, str):
            expected = schema.strip().lower()
            if expected in {"any", "*"}:
                return
            if expected == "string" and not isinstance(payload, str):
                raise ValidationError(f"Expected string at {path}, got {type(payload).__name__}.")
            if expected == "number" and (
                not isinstance(payload, (int, float)) or isinstance(payload, bool)
            ):
                raise ValidationError(f"Expected number at {path}, got {type(payload).__name__}.")
            if expected == "boolean" and not isinstance(payload, bool):
                raise ValidationError(f"Expected boolean at {path}, got {type(payload).__name__}.")
            if expected == "object" and not isinstance(payload, dict):
                raise ValidationError(f"Expected object at {path}, got {type(payload).__name__}.")
            if expected == "array" and not isinstance(payload, list):
                raise ValidationError(f"Expected array at {path}, got {type(payload).__name__}.")
            return

    def _validate_or_repair_text_chunk(
        self,
        raw_output: str,
        expected_page_start: int,
        expected_page_end: int,
    ) -> LLMPageChunk:
        try:
            payload = parse_llm_json(raw_output)
            return self._validate_chunk_payload(
                payload,
                expected_page_start=expected_page_start,
                expected_page_end=expected_page_end,
            )
        except Exception as first_error:
            repair_prompt = build_text_chunk_repair_prompt(
                raw_output=raw_output,
                expected_page_start=expected_page_start,
                expected_page_end=expected_page_end,
            )
            repaired_output = self.llm_client.complete_text(repair_prompt, temperature=0.0)

            try:
                repaired_payload = parse_llm_json(repaired_output)
                return self._validate_chunk_payload(
                    repaired_payload,
                    expected_page_start=expected_page_start,
                    expected_page_end=expected_page_end,
                )
            except Exception as second_error:
                raise ValidationError(
                    f"Failed to validate/repair text chunk {expected_page_start}-{expected_page_end}. "
                    f"Initial error: {first_error}; Repair error: {second_error}"
                ) from second_error

    def _validate_or_repair_text_page(self, raw_output: str, expected_page_number: int) -> LLMPage:
        try:
            payload = parse_llm_json(raw_output)
            return self._validate_page_payload(payload, expected_page_number=expected_page_number)
        except Exception as first_error:
            repair_prompt = build_text_repair_prompt(raw_output, expected_page_number)
            repaired_output = self.llm_client.complete_text(repair_prompt, temperature=0.0)

            try:
                repaired_payload = parse_llm_json(repaired_output)
                return self._validate_page_payload(
                    repaired_payload,
                    expected_page_number=expected_page_number,
                )
            except Exception as second_error:
                raise ValidationError(
                    f"Failed to validate/repair text page {expected_page_number}. "
                    f"Initial error: {first_error}; Repair error: {second_error}"
                ) from second_error

    def _validate_or_repair_ocr_batch(
        self,
        raw_output: str,
        expected_page_numbers: list[int],
    ) -> list[LLMPage]:
        try:
            payload = parse_llm_json(raw_output)
            return self._validate_ocr_batch_payload(payload, expected_page_numbers)
        except Exception as first_error:
            repair_prompt = build_ocr_repair_prompt(raw_output, expected_page_numbers)
            repaired_output = self.llm_client.complete_text(repair_prompt, temperature=0.0)

            try:
                repaired_payload = parse_llm_json(repaired_output)
                return self._validate_ocr_batch_payload(repaired_payload, expected_page_numbers)
            except Exception as second_error:
                raise ValidationError(
                    "Failed to validate/repair OCR batch "
                    f"{expected_page_numbers}. Initial error: {first_error}; "
                    f"Repair error: {second_error}"
                ) from second_error

    def _validate_page_payload(self, payload: Any, expected_page_number: int) -> LLMPage:
        try:
            llm_page = LLMPage.model_validate(payload)
        except PydanticValidationError as exc:
            raise ValidationError(
                f"Invalid page schema for page {expected_page_number}: {exc}"
            ) from exc

        if llm_page.page_number != expected_page_number:
            raise ValidationError(
                f"Expected page_number={expected_page_number}, got {llm_page.page_number}."
            )

        return llm_page

    def _validate_chunk_payload(
        self,
        payload: Any,
        expected_page_start: int,
        expected_page_end: int,
    ) -> LLMPageChunk:
        try:
            llm_chunk = LLMPageChunk.model_validate(payload)
        except PydanticValidationError as exc:
            raise ValidationError(
                f"Invalid chunk schema for {expected_page_start}-{expected_page_end}: {exc}"
            ) from exc

        if llm_chunk.page_start != expected_page_start or llm_chunk.page_end != expected_page_end:
            raise ValidationError(
                f"Expected page range {expected_page_start}-{expected_page_end}, "
                f"got {llm_chunk.page_start}-{llm_chunk.page_end}."
            )

        return llm_chunk

    def _validate_ocr_batch_payload(
        self,
        payload: Any,
        expected_page_numbers: list[int],
    ) -> list[LLMPage]:
        items: Any = payload
        if isinstance(payload, dict) and "pages" in payload:
            items = payload["pages"]

        if not isinstance(items, list):
            raise ValidationError("OCR batch output must be a JSON array of page objects.")

        parsed_pages: list[LLMPage] = []
        for item in items:
            try:
                parsed_pages.append(LLMPage.model_validate(item))
            except PydanticValidationError as exc:
                raise ValidationError(f"Invalid OCR page schema: {exc}") from exc

        expected = list(expected_page_numbers)
        returned = [page.page_number for page in parsed_pages]

        if len(returned) != len(expected):
            raise ValidationError(
                f"OCR batch returned {len(returned)} pages, expected {len(expected)} pages."
            )

        if set(returned) != set(expected):
            raise ValidationError(
                f"OCR batch page numbers mismatch. expected={expected}; returned={returned}"
            )

        page_map = {page.page_number: page for page in parsed_pages}
        return [page_map[page_number] for page_number in expected]

    @staticmethod
    def _to_parsed_page(llm_page: LLMPage) -> ParsedPage:
        elements = [
            Element(type=element.type, text=element.text, page_number=llm_page.page_number)
            for element in llm_page.elements
        ]
        return ParsedPage(
            page_number=llm_page.page_number,
            markdown=llm_page.markdown,
            elements=elements,
            tables=llm_page.tables,
        )

    @staticmethod
    def _to_parsed_chunk(llm_chunk: LLMPageChunk) -> ParsedPage:
        elements = [
            Element(type=element.type, text=element.text, page_number=llm_chunk.page_start)
            for element in llm_chunk.elements
        ]
        return ParsedPage(
            page_number=llm_chunk.page_start,
            page_end=llm_chunk.page_end,
            markdown=llm_chunk.markdown,
            elements=elements,
            tables=llm_chunk.tables,
        )

    @staticmethod
    def _looks_like_vision_unavailable(error_message: str) -> bool:
        message = error_message.lower()
        markers = [
            "vision",
            "image",
            "multimodal",
            "does not support",
            "unsupported",
            "cannot render",
        ]
        return any(marker in message for marker in markers)


_T = TypeVar("_T")


def _chunked(items: list[_T], size: int) -> list[list[_T]]:
    if size <= 0:
        raise ValueError("chunk size must be > 0")

    return [items[index : index + size] for index in range(0, len(items), size)]


def _build_chunk_text(chunk: list[tuple[int, str]]) -> str:
    lines: list[str] = []
    for page_number, text in chunk:
        lines.append(f"[PAGE {page_number}]")
        lines.append(text)
        lines.append("")
    return "\n".join(lines).strip()
