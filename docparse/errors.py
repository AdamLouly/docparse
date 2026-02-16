from __future__ import annotations


class DocParseError(Exception):
    """Base exception for docparse failures."""


class ProviderError(DocParseError):
    """Raised when provider calls fail."""


class ValidationError(DocParseError):
    """Raised when model output cannot be validated."""


class OCRUnavailableError(DocParseError):
    """Raised when OCR pipeline cannot run due to missing dependencies/capabilities."""
