"""Security helpers for safe file uploads, filenames, and input sanitization."""

import os
import re
from pathlib import Path
from typing import Tuple

ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent directory traversal and special character attacks."""
    basename = os.path.basename(filename)
    clean_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", basename)
    return clean_name or "uploaded_dataset"


def validate_file_upload(filename: str, content_length: int) -> Tuple[bool, str]:
    """Verify file extension and size constraints."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported file extension '{ext}'. Allowed types: {', '.join(sorted(ALLOWED_EXTENSIONS))}"

    if content_length > MAX_FILE_SIZE:
        return False, f"File size exceeds limit of {MAX_FILE_SIZE // (1024 * 1024)} MB."

    return True, "Valid"


def sanitize_user_input(text: str, max_length: int = 400) -> str:
    """Strip dangerous characters and bound length of user natural language queries."""
    if not text:
        return ""
    cleaned = re.sub(r"\s+", " ", text.strip())
    return cleaned[:max_length]
