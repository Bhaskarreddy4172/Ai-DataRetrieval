"""Universal Dataset Importer: Discovers, validates, normalizes, and ingests datasets into the database."""

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
from sqlalchemy.orm import Session

from app.config import BASE_DIR, settings
from app.database.connection import db_manager
from app.database.models import (
    DatasetColumnModel, DatasetModel, DatasetRelationshipModel,
    EntityAliasModel, EntityModel, StateDataModel, VillageDataModel,
    ValidationResultModel
)
from app.database.repositories import (
    DatasetRepository, EntityRepository, StateVillageRepository
)
from app.ingestion.column_mapper import column_mapper
from app.ingestion.csv_loader import csv_loader
from app.ingestion.excel_loader import excel_loader
from app.ingestion.normalizer import normalizer
from app.ingestion.schema_detector import schema_detector
from app.ingestion.validator import ingestion_validator
from app.dataset.alias_resolver import INDIAN_STATE_CODES
from app.utils.logger import logger


class DatasetImporter:
    """Orchestrates end-to-end ingestion from raw tabular files to SQL database."""

    def __init__(self):
        self.dataset_repo = DatasetRepository()
        self.entity_repo = EntityRepository()
        self.state_village_repo = StateVillageRepository()

    @staticmethod
    def compute_file_hash(file_path: Path) -> str:
        hasher = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception:
            return "unknown_hash"

    def load_file(self, file_path: Path) -> pd.DataFrame:
        suffix = file_path.suffix.lower()
        if suffix == ".csv":
            return csv_loader.load(file_path)
        elif suffix in [".xlsx", ".xls"]:
            return excel_loader.load(file_path)
        elif suffix == ".json":
            df = pd.read_json(file_path)
            df.columns = [str(c).strip() for c in df.columns]
            return df
        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def import_main_dataset(self, file_path: Path) -> Dict[str, Any]:
        """Ingests the main state-level dataset and registers states and capitals."""
        dataset_id = file_path.stem
        logger.info(f"Ingesting main dataset: {file_path.name}")
        df = self.load_file(file_path)

        # Standardize and normalize
        std_df = column_mapper.standardize_dataframe(df)
        clean_df = normalizer.normalize_dataframe(std_df)

        # Validate
        val_res = ingestion_validator.validate_dataset(clean_df, dataset_type="MAIN", dataset_name=dataset_id)

        # Schema detection
        columns_meta = schema_detector.analyze_dataframe(df)

        # Save dataset catalog
        file_hash = self.compute_file_hash(file_path)
        self.dataset_repo.upsert_dataset({
            "dataset_id": dataset_id,
            "dataset_name": file_path.stem.replace("_", " ").title(),
            "file_name": file_path.name,
            "file_path": str(file_path),
            "file_type": file_path.suffix.upper().lstrip("."),
            "dataset_type": "MAIN",
            "parent_entity": "Country",
            "child_entity": "State",
            "row_count": len(clean_df),
            "column_count": len(clean_df.columns),
            "content_hash": file_hash,
            "status": "READY" if val_res["status"] != "FAIL" else "FAILED",
        })
        self.dataset_repo.save_columns(dataset_id, columns_meta)

        # Ingest state data records
        states_records = []
        for _, row in clean_df.iterrows():
            state_val = str(row.get("state", "")).strip()
            capital_val = str(row.get("capital", "")).strip()
            code_val = str(row.get("state_code", "")).strip() if "state_code" in row else None
            if state_val:
                states_records.append({
                    "dataset_id": dataset_id,
                    "state": state_val,
                    "capital": capital_val,
                    "state_code": code_val,
                })
                # Register State entity
                self.entity_repo.upsert_entity(
                    entity_type="State",
                    name=state_val,
                    dataset_id=dataset_id,
                )
                # Register Capital entity
                if capital_val:
                    self.entity_repo.upsert_entity(
                        entity_type="Capital",
                        name=capital_val,
                        parent_entity_id=f"state_{state_val.lower().replace(' ', '_')}",
                        dataset_id=dataset_id,
                    )
                # Register state code alias if present
                if code_val:
                    self.entity_repo.add_alias(
                        entity_id=f"state_{state_val.lower().replace(' ', '_')}",
                        alias=code_val,
                        alias_type="CODE"
                    )

        # Add Indian state code aliases from alias resolver
        for code, full_name in INDIAN_STATE_CODES.items():
            self.entity_repo.add_alias(
                entity_id=f"state_{full_name.lower().replace(' ', '_')}",
                alias=code,
                alias_type="ABBREVIATION"
            )

        self.state_village_repo.save_state_records(dataset_id, states_records)
        return {"status": "SUCCESS", "dataset_id": dataset_id, "rows": len(states_records)}

    def import_child_dataset(
        self,
        file_path: Path,
        parent_dataset_id: str = "india_states_capitals_main"
    ) -> Dict[str, Any]:
        """Ingests a child village-level dataset across any state."""
        dataset_id = file_path.stem
        df = self.load_file(file_path)

        std_df = column_mapper.standardize_dataframe(df)
        clean_df = normalizer.normalize_dataframe(std_df)

        val_res = ingestion_validator.validate_dataset(clean_df, dataset_type="CHILD", dataset_name=dataset_id)
        columns_meta = schema_detector.analyze_dataframe(df)

        file_hash = self.compute_file_hash(file_path)
        self.dataset_repo.upsert_dataset({
            "dataset_id": dataset_id,
            "dataset_name": file_path.stem.replace("_", " ").title(),
            "file_name": file_path.name,
            "file_path": str(file_path),
            "file_type": file_path.suffix.upper().lstrip("."),
            "dataset_type": "CHILD",
            "parent_dataset_id": parent_dataset_id,
            "parent_entity": "State",
            "child_entity": "Village",
            "row_count": len(clean_df),
            "column_count": len(clean_df.columns),
            "content_hash": file_hash,
            "status": "READY" if val_res["status"] != "FAIL" else "FAILED",
        })
        self.dataset_repo.save_columns(dataset_id, columns_meta)

        # Save relationship
        self.dataset_repo.save_relationship({
            "relationship_id": f"{parent_dataset_id}_{dataset_id}",
            "parent_dataset_id": parent_dataset_id,
            "child_dataset_id": dataset_id,
            "parent_column": "State",
            "child_column": "State",
            "relationship_type": "ONE_TO_MANY"
        })

        # Save village data records
        village_records = []
        for _, row in clean_df.iterrows():
            village_name = str(row.get("village", "")).strip()
            state_val = str(row.get("state", "")).strip()
            capital_val = str(row.get("capital", "")).strip()

            if village_name:
                village_records.append({
                    "dataset_id": dataset_id,
                    "state": state_val,
                    "capital": capital_val,
                    "village": village_name,
                    "population": float(row.get("population", 0.0) or 0.0),
                    "no_of_males": float(row.get("no_of_males", 0.0) or 0.0),
                    "no_of_females": float(row.get("no_of_females", 0.0) or 0.0),
                    "literacy_rate_percent": float(row.get("literacy_rate_percent", 0.0) or 0.0),
                    "area_sq_km": float(row.get("area_sq_km", 0.0) or 0.0),
                    "households": float(row.get("households", 0.0) or 0.0),
                })
                # Register Village entity
                self.entity_repo.upsert_entity(
                    entity_type="Village",
                    name=village_name,
                    parent_entity_id=f"state_{state_val.lower().replace(' ', '_')}" if state_val else None,
                    dataset_id=dataset_id,
                )

        self.state_village_repo.save_village_records(dataset_id, village_records)
        return {"status": "SUCCESS", "dataset_id": dataset_id, "rows": len(village_records)}

    def import_all(self) -> Dict[str, Any]:
        """Automatically discovers and imports all main and child datasets."""
        results = {"main": [], "children": [], "total_rows": 0}

        # 1. Main datasets
        main_candidates = [
            BASE_DIR / "universal_dataset_demo" / "main_dataset" / "india_states_capitals_main.csv",
            settings.DATA_DIR / "main_dataset" / "india_states_capitals_main.csv",
            settings.DATA_DIR / "india_states_capitals_main.csv",
        ]
        main_found = None
        for cand in main_candidates:
            if cand.exists():
                main_found = cand
                res = self.import_main_dataset(cand)
                results["main"].append(res)
                break

        parent_id = main_found.stem if main_found else "india_states_capitals_main"

        # 2. Child datasets
        child_dirs = [
            BASE_DIR / "universal_dataset_demo" / "child_datasets",
            settings.DATA_DIR / "child_datasets",
        ]
        for cdir in child_dirs:
            if cdir.exists() and cdir.is_dir():
                for cfile in sorted(cdir.glob("*.csv")):
                    if "demographics" in cfile.name.lower():
                        continue  # Standalone state demographics summary
                    try:
                        res = self.import_child_dataset(cfile, parent_dataset_id=parent_id)
                        results["children"].append(res)
                        results["total_rows"] += res.get("rows", 0)
                    except Exception as e:
                        logger.error(f"Failed to ingest child dataset {cfile.name}: {e}")

        logger.info(f"Universal Ingestion complete: {len(results['children'])} child datasets, {results['total_rows']} rows.")
        return results


dataset_importer = DatasetImporter()
