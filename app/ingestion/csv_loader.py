"""CSV Loader: Robust CSV file reader with automatic encoding and delimiter detection."""

import csv
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
from app.utils.logger import logger


ENCODINGS_TO_TRY = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]


class CSVLoader:
    """Robust CSV Loader with fallback encodings and separator sniffing."""

    @staticmethod
    def detect_separator_and_encoding(file_path: Path) -> Tuple[str, str]:
        """Detect the delimiter and encoding of a CSV file."""
        chosen_encoding = "utf-8"
        sample_text = ""

        for enc in ENCODINGS_TO_TRY:
            try:
                with open(file_path, "r", encoding=enc) as f:
                    sample_text = f.read(4096)
                    chosen_encoding = enc
                    break
            except (UnicodeDecodeError, Exception):
                continue

        # Detect delimiter using csv.Sniffer or fallback
        try:
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(sample_text, delimiters=",\t;|")
            delimiter = dialect.delimiter
        except Exception:
            # Fallback heuristic
            counts = {",": sample_text.count(","), "\t": sample_text.count("\t"), ";": sample_text.count(";")}
            delimiter = max(counts, key=counts.get) if counts else ","
            if counts[delimiter] == 0:
                delimiter = ","

        return delimiter, chosen_encoding

    @classmethod
    def load(cls, file_path: Path) -> pd.DataFrame:
        """Load CSV into a clean DataFrame."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        delimiter, encoding = cls.detect_separator_and_encoding(file_path)

        try:
            df = pd.read_csv(file_path, sep=delimiter, encoding=encoding)
        except Exception as e:
            logger.warning(f"Failed to read {file_path} with {encoding}, falling back to latin-1: {e}")
            df = pd.read_csv(file_path, sep=delimiter, encoding="latin-1", on_bad_lines="skip")

        # Strip whitespace from column headers
        df.columns = [str(c).strip() for c in df.columns]
        return df


csv_loader = CSVLoader()
