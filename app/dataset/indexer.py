"""Dataset indexing engine: inverted value index, token index, and fast entity lookups."""

import os
import re
import threading
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from app.utils.logger import logger
from app.utils.fuzzy_match import fuzzy_match_value, LOCATION_ABBREVIATIONS
from app.utils.phonetic import PhoneticIndex, phonetic_match_score
from app.dataset.alias_resolver import entity_alias_resolver


class DatasetIndexer:
    """In-memory multi-level indexer for ultra-fast candidate value and entity lookups."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.column_uniques: Dict[str, List[Any]] = {}
        self.column_uniques_lower: Dict[str, Dict[str, Any]] = {}
        self.token_to_locations: Dict[str, List[Tuple[int, str]]] = {}
        self.numeric_summaries: Dict[str, Dict[str, float]] = {}
        self.phonetic_index: PhoneticIndex = PhoneticIndex()
        self.row_count: int = 0
        self.indexed_dataset_name: str = ""

        # Global Multi-Dataset Inverted Index Structures
        self.multi_dataset_tokens: Dict[str, List[Tuple[str, int, str]]] = {}
        self.multi_dataset_uniques: Dict[str, Dict[str, List[Any]]] = {}
        self.multi_dataset_uniques_lower: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.global_phonetic_index: PhoneticIndex = PhoneticIndex()
        self.indexed_datasets: Dict[str, str] = {}

    def _resolve_col_name(self, column: str) -> Optional[str]:
        """Case-insensitive column name resolution."""
        if not column:
            return None
        if column in self.column_uniques:
            return column
        col_low = column.strip().lower()
        for k in self.column_uniques:
            if k.lower() == col_low:
                return k
        return None

    def build_index(self, df: pd.DataFrame, dataset_name: str = "") -> None:
        """Construct full inverted index and column-level unique maps from DataFrame."""
        with self._lock:
            if df.empty:
                self.clear()
                return

            self.indexed_dataset_name = dataset_name
            self.row_count = len(df)
            self.column_uniques.clear()
            self.column_uniques_lower.clear()
            self.token_to_locations.clear()
            self.numeric_summaries.clear()
            self.phonetic_index = PhoneticIndex()

            # Filter out internal columns starting with '_' and cast column names to str
            active_cols: List[str] = [str(c).strip() for c in df.columns if not str(c).startswith("_")]

            # 1. Index columns
            for col_name in active_cols:
                raw_series = df[col_name]
                series: pd.Series = raw_series.iloc[:, 0] if isinstance(raw_series, pd.DataFrame) else raw_series
                valid_vals: List[Any] = series.dropna().unique().tolist()
                self.column_uniques[col_name] = valid_vals
                self.column_uniques_lower[col_name] = {str(v).strip().lower(): v for v in valid_vals if pd.notna(v)}

                # Index into phonetic index
                for v in valid_vals:
                    if isinstance(v, str) and not v.replace(".", "", 1).isdigit():
                        self.phonetic_index.add_term(v)

                # Numeric summary
                cleaned_str = series.astype(str).str.replace(r"[₹$,]", "", regex=True)
                converted = pd.to_numeric(cleaned_str, errors="coerce")
                if isinstance(converted, pd.Series):
                    num_s = converted.dropna()
                    if not num_s.empty and len(num_s) >= len(series) * 0.5:
                        num_arr = num_s.to_numpy(dtype=float)
                        if len(num_arr) > 0:
                            self.numeric_summaries[col_name] = {
                                "min": float(np.min(num_arr)),
                                "max": float(np.max(num_arr)),
                                "mean": float(np.mean(num_arr)),
                                "median": float(np.median(num_arr)),
                            }

            # 2. Build inverted token index
            for idx, (row_idx, row) in enumerate(df.iterrows()):
                # Robust integer row index extraction (handles string indices, custom indices, MultiIndex)
                int_row_idx: int = idx
                if "_internal_row_id" in row:
                    internal_val = row["_internal_row_id"]
                    if isinstance(internal_val, (int, float, np.integer, np.floating)):
                        int_row_idx = int(internal_val)
                    elif isinstance(internal_val, str) and internal_val.isdigit():
                        int_row_idx = int(internal_val)
                elif isinstance(row_idx, (int, float, np.integer, np.floating)):
                    int_row_idx = int(row_idx)
                elif isinstance(row_idx, str) and row_idx.isdigit():
                    int_row_idx = int(row_idx)

                for col_name in active_cols:
                    val = row[col_name]
                    if isinstance(val, (pd.Series, pd.DataFrame)):
                        continue
                    if pd.isna(val) is True:
                        continue
                    val_str = str(val).strip().lower()
                    words = re.findall(r"\b[\w-]+\b", val_str)
                    for w in words:
                        if len(w) >= 2:
                            if w not in self.token_to_locations:
                                self.token_to_locations[w] = []
                            if len(self.token_to_locations[w]) < 100:
                                self.token_to_locations[w].append((int_row_idx, col_name))

                    if 1 < len(words) <= 4:
                        phrase = " ".join(words)
                        if phrase not in self.token_to_locations:
                            self.token_to_locations[phrase] = []
                        if len(self.token_to_locations[phrase]) < 100:
                            self.token_to_locations[phrase].append((int_row_idx, col_name))

            # 3. Build dynamic alias index
            entity_alias_resolver.index_dataset(df)

            logger.info(f"Built index for '{dataset_name}': {len(self.token_to_locations)} tokens, {len(self.column_uniques)} columns.")

    def find_matching_value_in_column(self, candidate: Any, column: str) -> Optional[Any]:
        """Find exact, token substring, or fuzzy matching value within a specific column."""
        with self._lock:
            if candidate is None or not column:
                return None

            col_key = self._resolve_col_name(column)
            if not col_key or col_key not in self.column_uniques_lower:
                return None

            lower_map = self.column_uniques_lower[col_key]
            cand_str = str(candidate).strip()
            if not cand_str:
                return None
            cand_clean = cand_str.lower()
            # Clean trailing/leading non-semantic punctuation and possessives ('s)
            cand_clean = re.sub(r"^[\'\"?]+", "", cand_clean)
            cand_clean = re.sub(r"[?!.,;:\'\"]+$", "", cand_clean)
            cand_clean = re.sub(r"\'s$", "", cand_clean).strip()
            if not cand_clean:
                return None

            # 1. Exact match against indexed column values first (highest precision)
            if cand_clean in lower_map:
                return lower_map[cand_clean]

            # 2. Check general alias & abbreviation resolver (e.g. AP -> Andhra Pradesh if column is State)
            alias_res = entity_alias_resolver.resolve_alias(cand_str, available_columns=[col_key])
            if alias_res and alias_res.get("canonical_name"):
                alias_cand = alias_res["canonical_name"].strip().lower()
                if alias_cand in lower_map:
                    return lower_map[alias_cand]
                    
                    

            # 3. Check location abbreviation expansion (e.g. BLR -> Bengaluru if column is Capital/City)
            if cand_clean in LOCATION_ABBREVIATIONS:
                loc_cand = LOCATION_ABBREVIATIONS[cand_clean].strip().lower()
                if loc_cand in lower_map:
                    return lower_map[loc_cand]

            # 4. Whole-word / token boundary containment (e.g. "macbook" in "Apple MacBook Air")
            # Require minimum length 3 or boundary match to prevent short-token false positives
            if len(cand_clean) >= 3:
                for val_lower, original_val in lower_map.items():
                    if re.search(rf"\b{re.escape(cand_clean)}\b", val_lower):
                        return original_val
                    if len(val_lower) >= 3 and re.search(rf"\b{re.escape(val_lower)}\b", cand_clean):
                        return original_val

            # 5. Fuzzy match against column unique values
            fuzzy_val = fuzzy_match_value(cand_clean, list(lower_map.keys()))
            if fuzzy_val and fuzzy_val in lower_map:
                return lower_map[fuzzy_val]

            # 6. Phonetic match
            candidates = self.phonetic_index.get_candidates(cand_clean)
            best_p_val = None
            best_p_score = 0.0
            for cand in candidates:
                c_low = cand.strip().lower()
                if c_low in lower_map:
                    score = phonetic_match_score(cand_clean, c_low)
                    if score > best_p_score and score >= 0.78:
                        best_p_score = score
                        best_p_val = lower_map[c_low]
            if best_p_val is not None:
                return best_p_val

            return None

    def search_token_across_columns(self, token: Any) -> List[Tuple[int, str]]:
        """Find row indices and column names matching a token or phrase."""
        with self._lock:
            if token is None:
                return []
            tok_str = str(token).strip()
            if not tok_str:
                return []
            tok_clean = tok_str.lower()

            # 1. Exact match in token index
            if tok_clean in self.token_to_locations:
                return self.token_to_locations[tok_clean]

            # 2. Check location abbreviation expansion
            if tok_clean in LOCATION_ABBREVIATIONS:
                loc_tok = LOCATION_ABBREVIATIONS[tok_clean].lower()
                if loc_tok in self.token_to_locations:
                    return self.token_to_locations[loc_tok]

            # 3. Whole-word / boundary token match across indexed tokens
            if len(tok_clean) >= 3:
                for indexed_tok, locs in self.token_to_locations.items():
                    if re.search(rf"\b{re.escape(tok_clean)}\b", indexed_tok):
                        return locs
                    if len(indexed_tok) >= 3 and re.search(rf"\b{re.escape(indexed_tok)}\b", tok_clean):
                        return locs

            return []

    def get_column_values(self, column: str) -> List[Any]:
        """Retrieve unique values for a column with case-insensitive fallback."""
        with self._lock:
            if not column:
                return []
            col_key = self._resolve_col_name(column)
            if col_key:
                return self.column_uniques.get(col_key, [])
            return []

    def index_dataset(self, df: pd.DataFrame, dataset_name: str, dataset_id: str) -> None:
        """Incrementally index any dataset into the global inverted index without wiping active dataset."""
        with self._lock:
            if df.empty:
                return

            self.indexed_datasets[dataset_id] = dataset_name
            if dataset_id not in self.multi_dataset_uniques:
                self.multi_dataset_uniques[dataset_id] = {}
                self.multi_dataset_uniques_lower[dataset_id] = {}

            active_cols = [str(c).strip() for c in df.columns if not str(c).startswith("_")]
            for col_name in active_cols:
                raw_series = df[col_name]
                uniques = [u for u in raw_series.dropna().unique().tolist() if str(u).strip() != ""]
                self.multi_dataset_uniques[dataset_id][col_name] = uniques

                lower_map: Dict[str, Any] = {}
                for val in uniques:
                    val_str = str(val).strip()
                    lower_map[val_str.lower()] = val
                    if len(val_str) >= 2:
                        self.global_phonetic_index.add_term(val_str)
                self.multi_dataset_uniques_lower[dataset_id][col_name] = lower_map

                # Build tokens for text columns
                if raw_series.dtype == object or pd.api.types.is_string_dtype(raw_series):
                    for idx, val in raw_series.dropna().items():
                        row_int = int(idx) if isinstance(idx, int) else 0
                        words = re.findall(r"\b[a-zA-Z0-9_-]+\b", str(val).lower())
                        for word in set(words):
                            if len(word) >= 2:
                                if word not in self.multi_dataset_tokens:
                                    self.multi_dataset_tokens[word] = []
                                self.multi_dataset_tokens[word].append((dataset_id, row_int, col_name))

    def search_token_globally(self, token: Any) -> List[Tuple[str, int, str]]:
        """Find matching (dataset_id, row_id, col_name) globally across all indexed datasets."""
        with self._lock:
            if token is None:
                return []
            tok_clean = str(token).strip().lower()
            if not tok_clean:
                return []
            return list(self.multi_dataset_tokens.get(tok_clean, []))

    def find_candidate_in_any_dataset(self, text: str) -> List[Dict[str, Any]]:
        """Search across all indexed datasets for an entity or token with confidence score."""
        with self._lock:
            if not text:
                return []
            t_clean = text.strip().lower()
            results: List[Dict[str, Any]] = []

            # 1. Exact match in lower unique maps
            for ds_key, col_maps in self.multi_dataset_uniques_lower.items():
                for col_name, val_map in col_maps.items():
                    if t_clean in val_map:
                        results.append({
                            "dataset": ds_key,
                            "column": col_name,
                            "value": val_map[t_clean],
                            "match_type": "exact",
                            "confidence": 1.0
                        })

            # 2. Token match in multi_dataset_tokens
            if not results and t_clean in self.multi_dataset_tokens:
                seen_ds: Set[str] = set()
                for ds_key, row_idx, col_name in self.multi_dataset_tokens[t_clean]:
                    if ds_key not in seen_ds:
                        seen_ds.add(ds_key)
                        results.append({
                            "dataset": ds_key,
                            "column": col_name,
                            "row_id": row_idx,
                            "value": t_clean,
                            "match_type": "token",
                            "confidence": 0.85
                        })

            return results

    def clear(self) -> None:
        """Reset all in-memory indexes and caches."""
        with self._lock:
            self.column_uniques.clear()
            self.column_uniques_lower.clear()
            self.token_to_locations.clear()
            self.numeric_summaries.clear()
            self.phonetic_index = PhoneticIndex()
            self.row_count = 0
            self.indexed_dataset_name = ""


dataset_indexer = DatasetIndexer()
