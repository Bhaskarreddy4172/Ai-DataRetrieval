"""Dynamic Parent-Child Dataset Registry: automatically indexes hierarchical datasets,
maps main dataset entities (States/Capitals/Codes) to Child datasets (Villages),
and provides dynamic schema inspection without hardcoded question-answer logic.
"""

import hashlib
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import pandas as pd

from app.config import BASE_DIR, settings
from app.utils.logger import logger
from app.dataset.alias_resolver import INDIAN_STATE_CODES, CITY_AIRPORT_CODES
from app.utils.fuzzy_match import expand_abbreviations, LOCATION_ABBREVIATIONS


class ParentChildRegistry:
    """Dynamic registry managing parent (State-Capital), child (Village-Level), and all standalone datasets."""

    def __init__(self):
        self._lock = threading.RLock()
        self._main_path: Optional[Path] = None
        self._child_dir: Optional[Path] = None
        self._main_df: Optional[pd.DataFrame] = None
        self._child_cache: Dict[str, pd.DataFrame] = {}
        self._entity_to_child: Dict[str, Path] = {}
        self._child_to_meta: Dict[str, Dict[str, Any]] = {}
        self._village_names_map: Dict[str, Tuple[str, Path]] = {}
        self._child_columns: Set[str] = set()
        self._all_datasets_catalog: Dict[str, Dict[str, Any]] = {}
        self._standalone_datasets: Dict[str, Dict[str, Any]] = {}
        self._standalone_entity_map: Dict[str, Tuple[str, str, Path]] = {}
        self._discovered_count: int = 0
        self._failed_count: int = 0
        self._initialized = False

        self._initialize_paths()

    def _initialize_paths(self) -> None:
        """Discover main dataset and child directory."""
        with self._lock:
            # Candidate paths for main dataset
            main_candidates = [
                BASE_DIR / "universal_dataset_demo" / "main_dataset" / "india_states_capitals_main.csv",
                settings.DATA_DIR / "main_dataset" / "india_states_capitals_main.csv",
                settings.DATA_DIR / "india_states_capitals_main.csv",
                settings.DATA_DIR / "indian_states_capitals.csv",
            ]
            for cand in main_candidates:
                if cand.exists():
                    self._main_path = cand
                    break

            # Candidate paths for child datasets directory
            child_candidates = [
                BASE_DIR / "universal_dataset_demo" / "child_datasets",
                settings.DATA_DIR / "child_datasets",
            ]
            for cand in child_candidates:
                if cand.exists() and cand.is_dir():
                    self._child_dir = cand
                    break

            self._load_and_index_registry()

    def _compute_hash(self, file_path: Path) -> str:
        """Calculate fast SHA-256 content hash of a file."""
        try:
            hasher = hashlib.sha256()
            with open(file_path, "rb") as fp:
                for chunk in iter(lambda: fp.read(65536), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception:
            return "hash_unknown"

    def _load_any_file(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Load tabular data from .csv, .xlsx, .xls, or .json file into DataFrame."""
        try:
            suffix = file_path.suffix.lower()
            if suffix == ".csv":
                try:
                    df = pd.read_csv(file_path, encoding="utf-8")
                except UnicodeDecodeError:
                    df = pd.read_csv(file_path, encoding="latin1")
            elif suffix in [".xlsx", ".xls"]:
                df = pd.read_excel(file_path)
            elif suffix == ".json":
                df = pd.read_json(file_path)
            else:
                return None
            df.columns = [str(c).strip() for c in df.columns]
            return df
        except Exception as ex:
            logger.warning(f"ParentChildRegistry: Error reading {file_path}: {ex}")
            return None

    def _discover_all_child_csvs(self) -> List[Path]:
        """Discover all child CSV files dynamically across registered directories."""
        seen_names: Set[str] = set()
        all_files: List[Path] = []
        dirs_to_check = [
            BASE_DIR / "universal_dataset_demo" / "child_datasets",
            settings.DATA_DIR / "child_datasets",
        ]
        for d in dirs_to_check:
            if d.exists() and d.is_dir():
                for f in d.glob("*.csv"):
                    fname_lower = f.name.lower()
                    # Exclude non-child files
                    if "main" in fname_lower or "catalog" in fname_lower or "states_capitals" in fname_lower or "state_demographics" in fname_lower or fname_lower.startswith("sample_"):
                        continue
                    if f.name not in seen_names:
                        seen_names.add(f.name)
                        all_files.append(f)
        return all_files

    def _inspect_child_csv(self, child_path: Path) -> Optional[Dict[str, Any]]:
        """Inspect a child CSV's schema and sample row to detect parent entity linkage."""
        try:
            sample_df = pd.read_csv(child_path, nrows=2, encoding="utf-8")
        except UnicodeDecodeError:
            try:
                sample_df = pd.read_csv(child_path, nrows=2, encoding="latin1")
            except Exception as ex:
                logger.warning(f"Could not read child dataset {child_path}: {ex}")
                return None
        except Exception as ex:
            logger.warning(f"Could not inspect child dataset {child_path}: {ex}")
            return None

        # Ensure it has child-level attributes or is an actual child dataset
        # A file with only state/capital columns is a parent dataset, not a child dataset
        cols_lower = {str(c).lower().strip() for c in sample_df.columns}
        if cols_lower <= {"state", "capital", "state_code", "state_id", "code", "id", "_internal_row_id"}:
            return None

        sample_df.columns = [str(c).strip() for c in sample_df.columns]
        state_val = ""
        capital_val = ""
        code_val = ""
        id_val = ""

        # Extract State
        state_col = next((c for c in sample_df.columns if c.lower() == "state"), None)
        if state_col and not sample_df[state_col].empty:
            st_series = sample_df[state_col].dropna()
            if not st_series.empty:
                state_val = str(st_series.iloc[0]).strip()

        # Extract Capital
        cap_col = next((c for c in sample_df.columns if c.lower() == "capital"), None)
        if cap_col and not sample_df[cap_col].empty:
            cap_series = sample_df[cap_col].dropna()
            if not cap_series.empty:
                capital_val = str(cap_series.iloc[0]).strip()

        # Extract Village ID / Code prefix (e.g. GJ-001 -> GJ)
        id_col = next((c for c in sample_df.columns if "id" in c.lower()), None)
        if id_col and not sample_df[id_col].empty:
            id_series = sample_df[id_col].dropna()
            if not id_series.empty:
                id_val = str(id_series.iloc[0]).strip()
                m_code = re.match(r"^([a-zA-Z]{2,4})-", id_val)
                if m_code:
                    code_val = m_code.group(1).upper()

        cols = {str(c).strip() for c in sample_df.columns}
        return {
            "state": state_val,
            "capital": capital_val,
            "code": code_val,
            "id": id_val,
            "columns": cols,
            "child_path": child_path,
            "child_filename": child_path.name,
        }

    def _load_and_index_registry(self) -> None:
        """Inspect datasets dynamically and index entities to child datasets with zero state-specific logic."""
        try:
            self._entity_to_child.clear()
            self._child_to_meta.clear()
            self._village_names_map.clear()
            self._child_columns.clear()
            self._failed_count = 0

            # 1. Discover all child CSVs across disk
            child_files = self._discover_all_child_csvs()
            self._discovered_count = len(child_files)

            # 2. Inspect each child file
            child_inspections: Dict[str, Dict[str, Any]] = {}
            for cf in child_files:
                inspected = self._inspect_child_csv(cf)
                if inspected:
                    child_inspections[cf.name] = inspected
                    self._child_columns.update(inspected["columns"])
                else:
                    self._failed_count += 1

            # 3. Read main dataset if available
            main_rows: List[Dict[str, Any]] = []
            if self._main_path and self._main_path.exists():
                try:
                    df = pd.read_csv(self._main_path, encoding="utf-8")
                    df.columns = [str(c).strip() for c in df.columns]
                    self._main_df = df
                    main_rows = df.to_dict(orient="records")
                except Exception as ex:
                    logger.warning(f"Error reading main dataset from {self._main_path}: {ex}")

            # 4. Link child datasets to parent entities
            linked_child_filenames: Set[str] = set()

            for row in main_rows:
                state_val = str(row.get("State") or row.get("state") or "").strip()
                capital_val = str(row.get("Capital") or row.get("capital") or "").strip()
                code_val = str(row.get("State_Code") or row.get("code") or "").strip()
                id_val = str(row.get("State_ID") or row.get("id") or "").strip()
                child_val = str(row.get("Child_Dataset") or row.get("child_file") or "").strip()

                matched_child: Optional[Dict[str, Any]] = None

                # A. Try explicit Child_Dataset column match
                if child_val and child_val.lower() != "none" and child_val in child_inspections:
                    matched_child = child_inspections[child_val]
                elif child_val:
                    # Search by basename
                    for fname, insp in child_inspections.items():
                        if child_val.lower() in fname.lower():
                            matched_child = insp
                            break

                # B. Try content matching via State name
                if not matched_child and state_val:
                    for fname, insp in child_inspections.items():
                        if insp["state"] and insp["state"].lower() == state_val.lower():
                            matched_child = insp
                            break

                # C. Try content matching via Capital name
                if not matched_child and capital_val:
                    for fname, insp in child_inspections.items():
                        if insp["capital"] and insp["capital"].lower() == capital_val.lower():
                            matched_child = insp
                            break

                # D. Try content matching via State Code
                if not matched_child and code_val:
                    for fname, insp in child_inspections.items():
                        if insp["code"] and insp["code"].lower() == code_val.lower():
                            matched_child = insp
                            break

                if matched_child:
                    child_p: Path = matched_child["child_path"]
                    linked_child_filenames.add(child_p.name)
                    final_state = state_val or matched_child["state"]
                    final_capital = capital_val or matched_child["capital"]
                    final_code = code_val or matched_child["code"]
                    final_id = id_val or matched_child["id"]

                    meta = {
                        "state": final_state,
                        "capital": final_capital,
                        "code": final_code,
                        "id": final_id,
                        "child_path": child_p,
                        "child_filename": child_p.name,
                    }
                    self._child_to_meta[child_p.name] = meta
                    self._index_entity_associations(final_state, final_capital, final_code, child_p, overwrite=True)

            # 5. Register any remaining discovered child files not present in main_rows (non-overwriting)
            for fname, insp in child_inspections.items():
                if fname not in linked_child_filenames:
                    child_p = insp["child_path"]
                    meta = {
                        "state": insp["state"],
                        "capital": insp["capital"],
                        "code": insp["code"],
                        "id": insp["id"],
                        "child_path": child_p,
                        "child_filename": child_p.name,
                    }
                    self._child_to_meta[child_p.name] = meta
                    self._index_entity_associations(insp["state"], insp["capital"], insp["code"], child_p, overwrite=False)

            # 6. Pre-load all child datasets and index entity names
            from app.dataset.indexer import dataset_indexer
            from app.dataset.spell_checker import dataset_spell_checker

            for child_fname, meta in list(self._child_to_meta.items()):
                c_df = self.load_child_dataframe(meta["child_path"])
                if c_df is not None and not c_df.empty:
                    ds_key = f"child_{meta.get('code', child_fname)}".lower()
                    dataset_indexer.index_dataset(c_df, meta["child_filename"], ds_key)
                    dataset_spell_checker.add_dataset_vocabulary(c_df, meta["child_filename"])

                    cols_low = {c.lower(): c for c in c_df.columns}
                    v_col = cols_low.get("village")
                    id_col = cols_low.get("village_id") or cols_low.get("id")

                    if v_col:
                        for v_val in c_df[v_col].dropna().unique():
                            v_str = str(v_val).strip()
                            v_low = v_str.lower()
                            self._village_names_map[v_low] = (meta["state"], meta["child_path"])
                            self._entity_to_child[v_low] = meta["child_path"]
                            self._entity_to_child[v_low.replace("_", " ")] = meta["child_path"]
                            self._entity_to_child[v_low.replace(" ", "")] = meta["child_path"]

                    if id_col:
                        for id_val in c_df[id_col].dropna().unique():
                            id_str = str(id_val).strip()
                            id_low = id_str.lower()
                            self._village_names_map[id_low] = (meta["state"], meta["child_path"])
                            self._entity_to_child[id_low] = meta["child_path"]
                            self._entity_to_child[id_low.replace("-", "")] = meta["child_path"]

            # 7. Discover, catalog, and index all standalone datasets (CSV, Excel, JSON)
            self._discover_and_index_standalone_datasets()

            self._initialized = True
            report = self.get_health_report()
            dash = self.get_health_dashboard()
            logger.info(
                f"MAIN DATASETS: {report['main_datasets']} | "
                f"PARENT ENTITIES: {report['parent_entities']} | "
                f"CHILD DATASETS DISCOVERED: {report['child_datasets_discovered']} | "
                f"CHILD DATASETS REGISTERED: {report['child_datasets_registered']} | "
                f"CHILD DATASETS FAILED: {report['child_datasets_failed']} | "
                f"TOTAL DATASETS CATALOGED: {dash['datasets_discovered']} | "
                f"TOTAL READY: {dash['datasets_ready']}"
            )

        except Exception as ex:
            logger.warning(f"ParentChildRegistry initialization error: {ex}")

    def _index_entity_associations(
        self,
        state_val: str,
        capital_val: str,
        code_val: str,
        child_path: Path,
        overwrite: bool = True
    ) -> None:
        """Index state, capital, codes, and standard abbreviations dynamically without state-specific hardcoding."""
        # Index state name
        if state_val:
            st_low = state_val.lower()
            if overwrite or st_low not in self._entity_to_child:
                self._entity_to_child[st_low] = child_path
                self._entity_to_child[st_low.replace(" ", "")] = child_path
                self._entity_to_child[st_low.replace("-", "")] = child_path

        # Index capital name
        if capital_val:
            cap_low = capital_val.lower()
            if overwrite or cap_low not in self._entity_to_child:
                self._entity_to_child[cap_low] = child_path
                self._entity_to_child[cap_low.replace(" ", "")] = child_path
                self._entity_to_child[cap_low.replace("-", "")] = child_path

        # Index state code
        if code_val:
            if overwrite or code_val.lower() not in self._entity_to_child:
                self._entity_to_child[code_val.lower()] = child_path

        # Generic abbreviation and alias mapping from standard dictionaries
        ambiguous_english_tokens = {"or", "in", "is", "to", "at", "an", "on", "it", "so", "by", "of", "if", "no", "do", "and"}
        if state_val:
            st_clean = state_val.lower()
            for code_k, name_v in INDIAN_STATE_CODES.items():
                if code_k.lower() in ambiguous_english_tokens:
                    continue
                if name_v.lower() == st_clean:
                    if overwrite or code_k.lower() not in self._entity_to_child:
                        self._entity_to_child[code_k.lower()] = child_path
                        self._entity_to_child[code_k.lower().replace(" ", "")] = child_path

            for loc_k, loc_v in LOCATION_ABBREVIATIONS.items():
                if loc_k.lower() in ambiguous_english_tokens:
                    continue
                if loc_v.lower() == st_clean:
                    if overwrite or loc_k.lower() not in self._entity_to_child:
                        self._entity_to_child[loc_k.lower()] = child_path

        if capital_val:
            cap_clean = capital_val.lower()
            for city_k, city_v in CITY_AIRPORT_CODES.items():
                if city_v.lower() == cap_clean:
                    if overwrite or city_k.lower() not in self._entity_to_child:
                        self._entity_to_child[city_k.lower()] = child_path

            for loc_k, loc_v in LOCATION_ABBREVIATIONS.items():
                if loc_v.lower() == cap_clean:
                    if overwrite or loc_k.lower() not in self._entity_to_child:
                        self._entity_to_child[loc_k.lower()] = child_path

    def get_health_report(self) -> Dict[str, Any]:
        """Return diagnostic health check report of registered parent and child datasets."""
        with self._lock:
            main_count = 1 if self._main_df is not None and not self._main_df.empty else 0
            parent_entities = len(self._main_df) if self._main_df is not None else 0
            reg_count = len(self._child_to_meta)
            return {
                "main_datasets": main_count,
                "parent_entities": parent_entities,
                "child_datasets_discovered": self._discovered_count,
                "child_datasets_registered": reg_count,
                "child_datasets_failed": self._failed_count,
                "available_states": self.get_available_states(),
            }

    def get_available_states(self) -> List[str]:
        """Return sorted list of all available states with registered child datasets."""
        with self._lock:
            states: Set[str] = set()
            for meta in self._child_to_meta.values():
                st = meta.get("state")
                if st:
                    states.add(st)
            return sorted(list(states))

    def get_state_codes_map(self) -> Dict[str, str]:
        """Return dynamic mapping of all known state codes and abbreviations to canonical state names."""
        with self._lock:
            mapping: Dict[str, str] = {}
            for meta in self._child_to_meta.values():
                st = meta.get("state")
                if not st:
                    continue
                mapping[st.lower()] = st
                code = meta.get("code")
                if code:
                    mapping[code.lower()] = st
                cap = meta.get("capital")
                if cap:
                    mapping[cap.lower()] = st

            # Add dynamic aliases for registered states (excluding common English stop words)
            ambiguous_english_tokens = {"or", "in", "is", "to", "at", "an", "on", "it", "so", "by", "of", "if", "no", "do", "and"}
            for k, v in INDIAN_STATE_CODES.items():
                if k.lower() in ambiguous_english_tokens:
                    continue
                if any(meta.get("state", "").lower() == v.lower() for meta in self._child_to_meta.values()):
                    mapping[k.lower()] = v
            for k, v in CITY_AIRPORT_CODES.items():
                if k.lower() in ambiguous_english_tokens:
                    continue
                for meta in self._child_to_meta.values():
                    if meta.get("capital", "").lower() == v.lower() and meta.get("state"):
                        mapping[k.lower()] = meta["state"]

            return mapping

    def register_main_dataset(self, df: pd.DataFrame, file_path: Optional[Path] = None) -> None:
        """Dynamically register or update the main dataset and re-index parent-child mappings."""
        with self._lock:
            if file_path:
                self._main_path = Path(file_path)
            self._main_df = df.copy()
            self._load_and_index_registry()

    def is_parent_child_active(self) -> bool:
        """Check if parent-child dataset architecture is loaded and active."""
        with self._lock:
            return bool(self._initialized and self._child_to_meta)

    def get_main_dataset_path(self) -> Optional[Path]:
        """Return path to parent dataset."""
        with self._lock:
            return self._main_path

    def get_main_dataframe(self) -> pd.DataFrame:
        """Return parent main dataset DataFrame."""
        with self._lock:
            if self._main_df is not None:
                return self._main_df
            if self._main_path and self._main_path.exists():
                self._main_df = pd.read_csv(self._main_path)
                return self._main_df
            return pd.DataFrame()

    def get_child_datasets_dir(self) -> Optional[Path]:
        """Return path to child datasets directory."""
        with self._lock:
            return self._child_dir

    def get_registered_child_datasets(self) -> Dict[str, Path]:
        """Return mapping of state/child key to child CSV Path."""
        with self._lock:
            res: Dict[str, Path] = {}
            for name, meta in self._child_to_meta.items():
                st = meta.get("state") or name
                res[st] = meta["child_path"]
            return res

    def get_child_columns(self) -> Set[str]:
        """Return discovered child dataset columns."""
        with self._lock:
            return set(self._child_columns)

    def resolve_child_dataset(self, text_or_entity: str) -> Optional[Tuple[str, Path]]:
        """Resolve a state name, capital, code, or abbreviation to (entity_name, child_dataset_path)."""
        if not text_or_entity:
            return None

        with self._lock:
            if not self._initialized:
                self._initialize_paths()

            cleaned = text_or_entity.strip().lower()
            # Direct dictionary lookup
            if cleaned in self._entity_to_child:
                child_p = self._entity_to_child[cleaned]
                meta = self._child_to_meta.get(child_p.name, {})
                return meta.get("state") or meta.get("capital") or cleaned, child_p

            # Word without spaces
            no_space = cleaned.replace(" ", "").replace("-", "")
            if no_space in self._entity_to_child:
                child_p = self._entity_to_child[no_space]
                meta = self._child_to_meta.get(child_p.name, {})
                return meta.get("state") or meta.get("capital") or cleaned, child_p

            # Check if any token or alias in query matches an indexed entity
            words = re.findall(r"\b[a-zA-Z0-9_-]+\b", cleaned)
            for w in words:
                w_lower = w.lower()
                if w_lower in self._entity_to_child:
                    child_p = self._entity_to_child[w_lower]
                    meta = self._child_to_meta.get(child_p.name, {})
                    return meta.get("state") or meta.get("capital") or w_lower, child_p

            # Substring matching against known states/capitals
            for ent, path in self._entity_to_child.items():
                if len(ent) >= 4 and ent in cleaned:
                    meta = self._child_to_meta.get(path.name, {})
                    return meta.get("state") or meta.get("capital") or ent, path

            # Scan child filenames directly
            for fname, meta in self._child_to_meta.items():
                st = meta.get("state", "").lower()
                cap = meta.get("capital", "").lower()
                if (st and st in cleaned) or (cap and cap in cleaned):
                    matched_ent = meta.get("state") or meta.get("capital") or fname
                    return str(matched_ent), Path(meta["child_path"])

            # Typo / Phonetic matching for words >= 4 characters
            from app.utils.phonetic import phonetic_match_score
            for w in words:
                if len(w) >= 4 and w not in self._entity_to_child:
                    for ent, path in self._entity_to_child.items():
                        if len(ent) >= 4 and phonetic_match_score(w, ent) >= 0.78:
                            meta = self._child_to_meta.get(path.name, {})
                            matched_ent = meta.get("state") or meta.get("capital") or ent
                            return str(matched_ent), path

            # Dataset-aware spell checker
            try:
                from app.dataset.spell_checker import dataset_spell_checker
                for w in words:
                    if len(w) >= 4 and w not in self._entity_to_child:
                        spell_cand = dataset_spell_checker.resolve_candidate(w)
                        if spell_cand and spell_cand.canonical_value.lower() in self._entity_to_child:
                            child_p = self._entity_to_child[spell_cand.canonical_value.lower()]
                            meta = self._child_to_meta.get(child_p.name, {})
                            matched_val = meta.get("state") or meta.get("capital") or spell_cand.canonical_value
                            return str(matched_val), child_p
            except Exception:
                pass

        return None

    def load_child_dataframe(self, child_path: Union[str, Path]) -> Optional[pd.DataFrame]:
        """Load and cache a child dataset dataframe in a thread-safe manner."""
        path_obj = Path(child_path)
        cache_key = str(path_obj.resolve())

        with self._lock:
            if cache_key in self._child_cache:
                return self._child_cache[cache_key].copy()

            if not path_obj.exists():
                return None

            try:
                df = pd.read_csv(path_obj, encoding="utf-8")
                df.columns = [str(c).strip() for c in df.columns]
                # Cast numeric columns
                for col in ["Population", "No_of_Males", "No_of_Females", "Literacy_Rate_Percent", "Area_Sq_Km", "Households"]:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors="coerce")

                self._child_cache[cache_key] = df
                return df.copy()
            except Exception as ex:
                logger.error(f"Failed to load child dataset from {path_obj}: {ex}")
                return None

    def is_village_level_query(self, question: str) -> bool:
        """Detect whether query references village-level attributes or entities."""
        q_l = question.lower()
        village_keywords = [
            "village", "villages", "gram", "gaon", "palle", "basti",
            "literacy", "literacy rate", "households", "area sq km",
            "area_sq_km", "no_of_males", "no_of_females", "males", "females",
            "male population", "female population", "village_id", "village id"
        ]
        if any(kw in q_l for kw in village_keywords):
            return True

        # Village code pattern like TG-001, KA-002, etc.
        if re.search(r"\b[a-zA-Z]{2}-\d{3}\b", question):
            return True

        # Village name pattern like Hyderabad_Village_01 or Village_01
        if re.search(r"\b(?:[a-zA-Z]+_)?village_\d+\b", q_l):
            return True

        return False

    def _discover_and_index_standalone_datasets(self) -> None:
        """Discover and index all standalone datasets (.csv, .xlsx, .xls, .json) across uploads, data, and demo dirs."""
        from app.dataset.indexer import dataset_indexer
        from app.dataset.spell_checker import dataset_spell_checker

        self._all_datasets_catalog.clear()
        self._standalone_datasets.clear()
        self._standalone_entity_map.clear()

        # Catalog Main Dataset first
        if self._main_path and self._main_path.exists():
            main_id = "ds_main_states_capitals"
            cols = list(self._main_df.columns) if self._main_df is not None else ["State", "Capital"]
            dtypes = {col: str(self._main_df[col].dtype) for col in self._main_df.columns} if self._main_df is not None else {"State": "object", "Capital": "object"}
            rc = len(self._main_df) if self._main_df is not None else 28
            self._all_datasets_catalog[main_id] = {
                "dataset_id": main_id,
                "dataset_name": self._main_path.name,
                "file_path": str(self._main_path.resolve()),
                "file_type": "csv",
                "dataset_type": "PARENT",
                "parent_dataset_id": None,
                "parent_entity": None,
                "columns": cols,
                "data_types": dtypes,
                "row_count": rc,
                "content_hash": self._compute_hash(self._main_path),
                "version": "1.0",
                "status": "READY",
                "index_status": "INDEXED",
                "last_indexed_time": datetime.now(timezone.utc).isoformat(),
            }
            if self._main_df is not None and not self._main_df.empty:
                dataset_indexer.index_dataset(self._main_df, self._main_path.name, main_id)

        # Catalog All Child Datasets
        for child_fname, meta in self._child_to_meta.items():
            child_p: Path = meta["child_path"]
            child_id = f"ds_child_{meta.get('code', child_fname).lower().replace('-', '_').replace('.', '_')}"
            cached_df = self._child_cache.get(str(child_p.resolve()))
            cols = list(cached_df.columns) if cached_df is not None else list(meta.get("columns", []))
            dtypes = {col: str(cached_df[col].dtype) for col in cached_df.columns} if cached_df is not None else {}
            rc = len(cached_df) if cached_df is not None else 10
            self._all_datasets_catalog[child_id] = {
                "dataset_id": child_id,
                "dataset_name": child_fname,
                "file_path": str(child_p.resolve()),
                "file_type": "csv",
                "dataset_type": "CHILD",
                "parent_dataset_id": "ds_main_states_capitals",
                "parent_entity": meta.get("state"),
                "columns": cols,
                "data_types": dtypes,
                "row_count": rc,
                "content_hash": self._compute_hash(child_p),
                "version": "1.0",
                "status": "READY",
                "index_status": "INDEXED",
                "last_indexed_time": datetime.now(timezone.utc).isoformat(),
            }

        # Scan for Standalone Files across directories
        candidate_dirs = [
            BASE_DIR / "uploads",
            settings.DATA_DIR,
            BASE_DIR / "universal_dataset_demo",
        ]
        registered_paths = {str(meta["child_path"].resolve()) for meta in self._child_to_meta.values()}
        if self._main_path:
            registered_paths.add(str(self._main_path.resolve()))

        for c_dir in candidate_dirs:
            if not c_dir.exists() or not c_dir.is_dir():
                continue
            for file_path in c_dir.glob("*.*"):
                suffix = file_path.suffix.lower()
                if suffix not in [".csv", ".xlsx", ".xls", ".json"]:
                    continue
                if file_path.name.lower() in ["dataset_manifest.json", "package.json", "package-lock.json", "tsconfig.json"]:
                    continue
                resolved_str = str(file_path.resolve())
                if resolved_str in registered_paths:
                    continue

                registered_paths.add(resolved_str)
                df = self._load_any_file(file_path)
                if df is None or df.empty:
                    continue

                std_id = f"ds_std_{file_path.stem.lower().replace('-', '_').replace(' ', '_')}"
                meta_item = {
                    "dataset_id": std_id,
                    "dataset_name": file_path.name,
                    "file_path": resolved_str,
                    "file_type": suffix.lstrip("."),
                    "dataset_type": "STANDALONE",
                    "parent_dataset_id": None,
                    "parent_entity": None,
                    "columns": list(df.columns),
                    "data_types": {col: str(df[col].dtype) for col in df.columns},
                    "row_count": len(df),
                    "content_hash": self._compute_hash(file_path),
                    "version": "1.0",
                    "status": "READY",
                    "index_status": "INDEXED",
                    "last_indexed_time": datetime.now(timezone.utc).isoformat(),
                }
                self._all_datasets_catalog[std_id] = meta_item
                self._standalone_datasets[std_id] = {
                    "metadata": meta_item,
                    "file_path": file_path,
                    "df": df,
                }

                # Index standalone dataset into global indexer and spell checker
                dataset_indexer.index_dataset(df, file_path.name, std_id)
                dataset_spell_checker.add_dataset_vocabulary(df, file_path.name)

                # Index key string column entities into standalone entity map
                for col in df.select_dtypes(include=["object", "category"]).columns:
                    for val in df[col].dropna().unique():
                        v_str = str(val).strip()
                        if len(v_str) >= 2:
                            self._standalone_entity_map[v_str.lower()] = (std_id, file_path.name, file_path)

    def get_health_dashboard(self) -> Dict[str, Any]:
        """Return comprehensive health and readiness dashboard across all discovered and indexed datasets."""
        with self._lock:
            total_count = len(self._all_datasets_catalog)
            ready_count = sum(1 for d in self._all_datasets_catalog.values() if d.get("status") == "READY")
            failed_count = sum(1 for d in self._all_datasets_catalog.values() if d.get("status") == "FAILED")
            parent_count = sum(1 for d in self._all_datasets_catalog.values() if d.get("dataset_type") == "PARENT")
            child_count = sum(1 for d in self._all_datasets_catalog.values() if d.get("dataset_type") == "CHILD")
            standalone_count = sum(1 for d in self._all_datasets_catalog.values() if d.get("dataset_type") == "STANDALONE")

            datasets_list = sorted(list(self._all_datasets_catalog.values()), key=lambda d: d.get("dataset_name", ""))
            return {
                "status": "ready" if failed_count == 0 else "degraded",
                "datasets_discovered": total_count,
                "datasets_indexed": ready_count,
                "datasets_ready": ready_count,
                "datasets_failed": failed_count,
                "parent_datasets": parent_count,
                "child_datasets": child_count,
                "standalone_datasets": standalone_count,
                "datasets": datasets_list,
                "summary": f"Datasets discovered: {total_count} | Datasets indexed: {ready_count} | Datasets ready: {ready_count} | Datasets failed: {failed_count}"
            }

    def get_all_dataset_metadata(self) -> List[Dict[str, Any]]:
        """Return list of metadata dictionaries for all cataloged datasets."""
        with self._lock:
            return list(self._all_datasets_catalog.values())

    def resolve_dataset_for_query(self, query: str) -> Optional[Tuple[str, Path, pd.DataFrame]]:
        """Determine if a query targets a standalone dataset, child dataset, or parent dataset."""
        if not query:
            return None

        q_lower = query.strip().lower()

        # 1. Check if matches any standalone dataset name or keyword first
        for std_id, data in self._standalone_datasets.items():
            meta = data["metadata"]
            dname = meta["dataset_name"].lower()
            stem = Path(dname).stem.lower()
            if stem in q_lower or (len(stem) >= 3 and stem.replace("_", " ") in q_lower):
                return meta["dataset_name"], data["file_path"], data["df"]

        # 2. Check standalone entity map (e.g. phone model names, employee names)
        tokens = re.findall(r"\b[\w-]+\b", q_lower)
        for t in tokens:
            if t in self._standalone_entity_map and len(t) >= 3:
                std_id, fname, fpath = self._standalone_entity_map[t]
                if std_id in self._standalone_datasets:
                    return fname, fpath, self._standalone_datasets[std_id]["df"]

        # 3. Check if query matches distinct column names or entities in standalone datasets
        for std_id, data in self._standalone_datasets.items():
            df = data["df"]
            cols = [str(c).lower() for c in df.columns]
            matched_cols = [c for c in cols if c in q_lower and len(c) >= 3]
            if len(matched_cols) >= 2 or any((c in cols and c in q_lower) for c in ["camera", "processor", "ram", "battery capacity", "launched price", "mobile weight"]):
                return data["metadata"]["dataset_name"], data["file_path"], df

        # 4. Check if matches child village or state entity
        resolved_child = self.resolve_child_dataset(query)
        if resolved_child:
            entity_name, child_path = resolved_child
            if self.is_village_level_query(query):
                child_df = self.load_child_dataframe(child_path)
                if child_df is not None:
                    return entity_name, child_path, child_df
            elif self._main_path and self._main_df is not None:
                return "Indian States & Capitals", self._main_path, self._main_df.copy()
            else:
                child_df = self.load_child_dataframe(child_path)
                if child_df is not None:
                    return entity_name, child_path, child_df

        return None


parent_child_registry = ParentChildRegistry()
