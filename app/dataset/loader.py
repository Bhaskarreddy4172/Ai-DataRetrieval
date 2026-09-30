import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
from app.config import settings
from app.dataset.indexer import dataset_indexer
from app.dataset.profiler import dataset_profiler
from app.dataset.relationship_engine import relationship_engine
from app.dataset.schema import schema_intelligence
from app.dataset.validator import validate_dataframe, validate_dataset_file
from app.utils.logger import logger


SAMPLE_DATASETS: Dict[str, Dict[str, Any]] = {
    "customers": {
        "id": "customers",
        "title": "Honda Customer After-Sales",
        "file": "sample_customer_data.xlsx",
        "description": "100 after-sales service records with vehicle models, service costs, and warranty statuses.",
        "type": "excel",
    },
    "states": {
        "id": "states",
        "title": "Indian States & Capitals",
        "file": "indian_states_capitals.csv",
        "description": "28 Indian states with official capital cities.",
        "type": "csv",
    },
    "products": {
        "id": "products",
        "title": "E-Commerce Electronics Catalog",
        "file": "sample_products.csv",
        "description": "Consumer electronics with categories, prices, stock levels, and ratings.",
        "type": "csv",
    },
    "employees": {
        "id": "employees",
        "title": "Corporate Employee Directory",
        "file": "sample_employees.xlsx",
        "description": "80 corporate employees across departments, designations, salaries, and performance scores.",
        "type": "excel",
    }
}


class DatasetLoader:
    """Manages active in-memory dataset, multi-sheet Excel workbooks, indexing, and schema intelligence."""

    def __init__(self, initial_path: Optional[str] = None):
        self._lock = threading.RLock()
        self._current_df: Optional[pd.DataFrame] = None
        self._dataset_name: str = "None"
        self._file_path: Optional[Path] = None
        self._profile: Optional[Dict[str, Any]] = None
        self._relationships: Optional[Dict[str, Any]] = None
        self._schema_intel: Optional[Dict[str, Any]] = None
        self._sheet_names: List[str] = []
        self._active_sheet: Optional[str] = None
        self._dataset_hash: str = ""

        # Load initial default dataset
        init_file = Path(initial_path or settings.DEFAULT_DATASET)
        if not init_file.exists():
            for sample_key in ["customers", "states"]:
                cand = settings.DATA_DIR / SAMPLE_DATASETS[sample_key]["file"]
                if cand.exists():
                    init_file = cand
                    break

        if init_file.exists():
            self.load_dataset(init_file)

    @property
    def dataframe(self) -> pd.DataFrame:
        with self._lock:
            if self._current_df is None:
                return pd.DataFrame()
            return self._current_df

    @dataframe.setter
    def dataframe(self, df: pd.DataFrame) -> None:
        with self._lock:
            self._current_df = df
            if df is not None and not df.empty:
                try:
                    from app.dataset.spell_checker import dataset_spell_checker
                    dataset_spell_checker.build_vocabulary(df, self._dataset_name, self._dataset_hash)
                except Exception:
                    pass

    @property
    def dataset_name(self) -> str:
        with self._lock:
            return self._dataset_name

    @dataset_name.setter
    def dataset_name(self, name: str) -> None:
        with self._lock:
            self._dataset_name = name

    @property
    def sheet_names(self) -> List[str]:
        with self._lock:
            return list(self._sheet_names)

    @property
    def active_sheet(self) -> Optional[str]:
        with self._lock:
            return self._active_sheet

    @property
    def profile(self) -> Dict[str, Any]:
        with self._lock:
            if self._profile is None and self._current_df is not None:
                self._profile = dataset_profiler.profile(self._current_df, self._dataset_name, self._file_path)
            return self._profile or {}

    @property
    def relationships(self) -> Dict[str, Any]:
        with self._lock:
            if self._relationships is None and self._current_df is not None:
                self._relationships = relationship_engine.analyze(self._current_df)
            return self._relationships or {}

    @property
    def schema_intelligence(self) -> Dict[str, Any]:
        with self._lock:
            if self._schema_intel is None and self._current_df is not None:
                self._schema_intel = schema_intelligence.build_schema(self._current_df, self._dataset_name)
            return self._schema_intel or {}

    def load_dataset(
        self,
        file_path: Union[Path, str],
        custom_name: Optional[str] = None,
        sheet_name: Optional[str] = None
    ) -> Tuple[bool, str]:
        """Load CSV or multi-sheet Excel file, build in-memory indexes, and compute deep schema intelligence."""
        with self._lock:
            file_path = Path(file_path)
            val_file, err_msg = validate_dataset_file(file_path)
            if not val_file:
                logger.error(err_msg)
                return False, err_msg

            try:
                ext = file_path.suffix.lower()
                self._sheet_names = []
                self._active_sheet = None

                if ext == ".csv":
                    try:
                        df = pd.read_csv(file_path, encoding="utf-8")
                    except UnicodeDecodeError:
                        df = pd.read_csv(file_path, encoding="latin1")
                elif ext in {".xlsx", ".xls"}:
                    excel_file = pd.ExcelFile(file_path)
                    self._sheet_names = excel_file.sheet_names
                    selected_sheet = sheet_name or (self._sheet_names[0] if self._sheet_names else 0)
                    self._active_sheet = str(selected_sheet)
                    df = pd.read_excel(file_path, sheet_name=selected_sheet)
                elif ext == ".json":
                    with open(file_path, "r", encoding="utf-8") as jf:
                        j_data = json.load(jf)
                    if isinstance(j_data, list):
                        df = pd.DataFrame(j_data)
                    elif isinstance(j_data, dict):
                        # Check if dict of lists or dict with a data key
                        if any(isinstance(v, list) for v in j_data.values()):
                            for k, v in j_data.items():
                                if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                                    df = pd.DataFrame(v)
                                    break
                            else:
                                df = pd.DataFrame(j_data)
                        else:
                            df = pd.DataFrame([j_data])
                    else:
                        return False, "Unsupported JSON structure; expected array of records or dictionary."
                else:
                    return False, f"Unsupported format: {ext}"

                val_df, df_err = validate_dataframe(df)
                if not val_df:
                    return False, df_err

                # Clean column names (strip whitespace)
                df.columns = [str(c).strip() for c in df.columns]

                # Strip string values
                for col in df.select_dtypes(include=["object"]).columns:
                    df[col] = df[col].astype(str).str.strip()

                # Assign stable internal row identifier
                df["_internal_row_id"] = range(len(df))

                self._current_df = df
                self._file_path = file_path
                self._dataset_name = custom_name or file_path.name
                self._canonical_rows = df.to_dict(orient="records")
                self._dataset_hash = hashlib.sha256(
                    f"{self._dataset_name}_{len(df)}_{sorted([str(c) for c in df.columns])}".encode("utf-8")
                ).hexdigest()[:16]

                # Reset conversation context when a new dataset is loaded
                try:
                    from app.conversation.context import conversation_manager
                    conversation_manager.clear_all()
                except Exception:
                    pass

                # Clear query cache on new dataset version
                try:
                    from app.query.cache import query_cache
                    query_cache.clear()
                except Exception:
                    pass

                # Profiling & deep schema intelligence
                self._profile = dataset_profiler.profile(self._current_df, self._dataset_name, self._file_path)
                self._schema_intel = schema_intelligence.build_schema(self._current_df, self._dataset_name)

                # Inverted indexing & relationship analysis
                dataset_indexer.build_index(self._current_df, self._dataset_name)
                self._relationships = relationship_engine.analyze(self._current_df)

                # Universal dataset-aware spell checker vocabulary
                try:
                    from app.dataset.spell_checker import dataset_spell_checker
                    dataset_spell_checker.build_vocabulary(self._current_df, self._dataset_name, self._dataset_hash)
                except Exception as ex:
                    logger.warning(f"Spell checker vocabulary build failed: {ex}")

                # Hybrid vector embeddings index
                try:
                    from app.dataset.vectorstore import dataset_vector_store
                    dataset_vector_store.build_index(self._current_df, self._dataset_name, self._dataset_hash)
                except Exception as ex:
                    logger.warning(f"Vector store indexing failed: {ex}")

                # Universal ParentChildRegistry update if dataset has state/capital/child columns
                try:
                    from app.dataset.registry import parent_child_registry
                    cols_lower = [str(c).lower() for c in df.columns]
                    if "state" in cols_lower or "capital" in cols_lower or "child_dataset" in cols_lower:
                        parent_child_registry.register_main_dataset(self._current_df, self._file_path)
                except Exception as ex:
                    logger.warning(f"ParentChildRegistry update failed: {ex}")

                logger.info(f"Loaded active dataset '{self._dataset_name}' ({len(df)} rows, {len(df.columns)} cols).")
                return True, f"Successfully loaded '{self._dataset_name}'."

            except Exception as e:
                msg = f"Failed to read dataset: {str(e)}"
                logger.error(msg)
                return False, msg

    def switch_sheet(self, sheet_name: str) -> Tuple[bool, str]:
        """Switch active sheet in multi-sheet Excel workbook."""
        with self._lock:
            if not self._file_path or not self._file_path.exists():
                return False, "No active Excel workbook loaded."
            if sheet_name not in self._sheet_names:
                return False, f"Sheet '{sheet_name}' not found. Available sheets: {self._sheet_names}"

            return self.load_dataset(self._file_path, custom_name=self._dataset_name, sheet_name=sheet_name)

    def load_sample(self, sample_id: str) -> Tuple[bool, str]:
        """Load one of the predefined sample datasets by key ('customers', 'states', 'products', 'employees')."""
        with self._lock:
            if sample_id not in SAMPLE_DATASETS:
                return False, f"Sample '{sample_id}' not found. Available: {list(SAMPLE_DATASETS.keys())}"

            sample_meta = SAMPLE_DATASETS[sample_id]
            file_path = settings.DATA_DIR / sample_meta["file"]
            if not file_path.exists():
                file_path = settings.UPLOADS_DIR / sample_meta["file"]
            if not file_path.exists():
                return False, f"Sample file '{file_path}' does not exist on disk."

            return self.load_dataset(file_path, custom_name=sample_meta["title"])

    def get_available_samples(self) -> List[Dict[str, Any]]:
        """List all predefined sample datasets."""
        res = []
        for sid, meta in SAMPLE_DATASETS.items():
            p = settings.DATA_DIR / meta["file"]
            if not p.exists():
                p = settings.UPLOADS_DIR / meta["file"]
            res.append({
                **meta,
                "exists": p.exists(),
                "is_active": self._dataset_name == meta["title"]
            })
        return res

    @property
    def canonical_rows(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(getattr(self, "_canonical_rows", []))

    @property
    def dataset_hash(self) -> str:
        with self._lock:
            return getattr(self, "_dataset_hash", "default_hash")

    def get_columns(self) -> List[str]:
        with self._lock:
            if self._current_df is not None:
                return [c for c in self._current_df.columns if c != "_internal_row_id"]
            return []

    def get_child_dataset_for_entity(self, entity_name: str) -> Optional[pd.DataFrame]:
        """Dynamically retrieve child dataset DataFrame for a given state or capital without evicting main."""
        from app.dataset.registry import parent_child_registry
        resolved = parent_child_registry.resolve_child_dataset(entity_name)
        if resolved:
            _, child_path = resolved
            return parent_child_registry.load_child_dataframe(child_path)
        return None


dataset_loader = DatasetLoader()

