from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class DiskCache:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def hash_bytes(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def hash_prompt(prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def build_key(self, file_hash: str, page_number: int, mode: str, model: str, prompt: str) -> str:
        prompt_hash = self.hash_prompt(prompt)
        base = f"{file_hash}|{page_number}|{mode}|{model}|{prompt_hash}"
        return hashlib.sha256(base.encode("utf-8")).hexdigest()

    def get(self, key: str) -> dict[str, Any] | None:
        cache_path = self.cache_dir / f"{key}.json"
        if not cache_path.exists():
            return None

        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def set(self, key: str, value: dict[str, Any]) -> None:
        cache_path = self.cache_dir / f"{key}.json"
        cache_path.write_text(json.dumps(value, ensure_ascii=True, indent=2), encoding="utf-8")
