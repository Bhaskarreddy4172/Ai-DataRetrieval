"""Excel Loader: Reads .xlsx and .xls tabular workbooks."""

from pathlib import Path
from typing import Optional, Union
import pandas as pd
from app.utils.logger import logger


class ExcelLoader:
    """Loads spreadsheet data from Excel workbooks."""

    @classmethod
    def load(cls, file_path: Path, sheet_name: Optional[Union[str, int]] = 0) -> pd.DataFrame:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name)
        except Exception as e:
            logger.error(f"Error loading Excel file {file_path}: {e}")
            raise

        df.columns = [str(c).strip() for c in df.columns]
        return df


excel_loader = ExcelLoader()
