"""Dataset Registry: Dynamic registry for discovering, inspecting, and managing tabular datasets."""

from typing import Any, Dict, List, Optional
from app.database.repositories import dataset_repo
from app.utils.logger import logger


class DatasetRegistry:
    """Provides high-level dataset querying, column schemas, and metadata inspection."""

    def __init__(self):
        self.repo = dataset_repo

    def get_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        ds = self.repo.get_dataset(dataset_id)
        if not ds:
            return None
        return {
            "dataset_id": ds.dataset_id,
            "dataset_name": ds.dataset_name,
            "file_name": ds.file_name,
            "file_path": ds.file_path,
            "file_type": ds.file_type,
            "dataset_type": ds.dataset_type,
            "parent_entity": ds.parent_entity,
            "child_entity": ds.child_entity,
            "row_count": ds.row_count,
            "column_count": ds.column_count,
            "status": ds.status,
        }

    def list_datasets(self, status: Optional[str] = "READY") -> List[Dict[str, Any]]:
        return self.repo.list_datasets(status=status)

    def get_columns(self, dataset_id: str) -> List[Dict[str, Any]]:
        return self.repo.get_columns(dataset_id)

    def find_datasets_with_column(self, col_name: str) -> List[str]:
        """Find all datasets containing a given normalized column name."""
        all_ds = self.list_datasets()
        matching = []
        target = col_name.strip().lower()
        for ds in all_ds:
            cols = self.get_columns(ds["dataset_id"])
            if any(c["normalized_name"] == target for c in cols):
                matching.append(ds["dataset_id"])
        return matching


dataset_registry = DatasetRegistry()
