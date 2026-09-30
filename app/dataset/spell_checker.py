"""Universal Dataset-Aware Spell Checker & Typo Resolution Engine.

Implements a 14-stage deterministic resolution pipeline:
1. Exact match
2. Case-insensitive match
3. Unicode normalization (NFKD)
4. Whitespace normalization (extra spaces, missing spaces)
5. Punctuation normalization (hyphens, underscores, slashes)
6. Compact form (alphanumerics only, e.g. "westbengal" -> "West Bengal")
7. Known alias & shortcut resolution (AP, TG, WB, HYD, BLR, etc.)
8. Abbreviation expansion
9. Edit-distance / Levenshtein (deletion, insertion, substitution, transposition)
10. RapidFuzz multi-metric matching (ratio, WRatio, token_sort_ratio, token_set_ratio)
11. Phonetic matching (Double Metaphone, Soundex, Indian phonetics)
12. Semantic matching (column synonyms and ontology)
13. Dataset context (entity-type filtering: STATE, CITY, PERSON, PRODUCT, etc.)
14. Confidence validation and ambiguity detection.

Strict rule: The client dataset is the single source of truth; original values are never modified.
"""

import re
import threading
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd
from rapidfuzz import fuzz, distance

from app.dataset.alias_resolver import entity_alias_resolver
from app.utils.phonetic import PhoneticIndex, double_metaphone, soundex, normalize_indian_phonetics, phonetic_match_score
from app.utils.logger import logger


@dataclass
class SpellingCorrection:
    """Detailed result of a spelling/typo resolution."""
    input_text: str
    corrected_value: str
    column_name: Optional[str] = None
    confidence: float = 1.0
    methods: List[str] = field(default_factory=list)
    is_ambiguous: bool = False
    candidates: List[str] = field(default_factory=list)
    entity_type: Optional[str] = None

    @property
    def canonical_value(self) -> str:
        return self.corrected_value

    @property
    def column(self) -> Optional[str]:
        return self.column_name


class ColumnMatch(str):
    """String subclass representing resolved column name with metadata."""
    corrected_value: str
    column_name: str
    canonical_value: str
    confidence: float
    methods: List[str]

    def __new__(cls, val: str, confidence: float = 1.0, method: str = "exact"):
        obj = str.__new__(cls, val)
        obj.corrected_value = val
        obj.column_name = val
        obj.canonical_value = val
        obj.confidence = confidence
        obj.methods = [method]
        return obj

    def to_spelling_correction(self, original_token: str) -> SpellingCorrection:
        return SpellingCorrection(
            input_text=original_token,
            corrected_value=self.corrected_value,
            column_name=self.column_name,
            confidence=self.confidence,
            methods=self.methods,
            entity_type="COLUMN"
        )


class FullQuestionCorrection(dict):
    """Container for question correction result supporting both dict and tuple access."""
    def __init__(self, corrected_question: str, replacements: List[Any]):
        super().__init__(
            corrected_question=corrected_question,
            replacements=replacements
        )
        self.corrected_question = corrected_question
        self.replacements = replacements

    def __iter__(self):
        return iter((self.corrected_question, self.replacements))

    def __getitem__(self, item):
        if isinstance(item, int):
            return (self.corrected_question, self.replacements)[item]
        return super().__getitem__(item)


class DatasetSpellChecker:
    """Dataset-aware spell checker and typo resolver with caching, ID protection, and context awareness."""

    # Common English / question words that should be normalized if misspelled
    QUERY_WORD_CORRECTIONS: Dict[str, str] = {
        "higest": "highest",
        "hihgest": "highest",
        "heighest": "highest",
        "hightest": "highest",
        "lowst": "lowest",
        "lowset": "lowest",
        "salry": "salary",
        "slary": "salary",
        "sallary": "salary",
        "captial": "capital",
        "capitl": "capital",
        "captal": "capital",
        "cpaital": "capital",
        "stat": "state",
        "staet": "state",
        "dpartmnt": "department",
        "depatment": "department",
        "dept": "department",
        "designtion": "designation",
        "desgnation": "designation",
        "employes": "employees",
        "employe": "employee",
        "empolyee": "employee",
        "empolyees": "employees",
        "customr": "customer",
        "custmer": "customer",
        "prodcut": "product",
        "prduct": "product",
        "prdouct": "product",
        "avrg": "average",
        "avrage": "average",
        "averge": "average",
        "medan": "median",
        "totl": "total",
        "whre": "where",
        "wich": "which",
        "whcih": "which",
        "wat": "what",
        "waht": "what",
        "erans": "earns",
        "eraning": "earning",
        "makse": "makes",
        "workin": "working",
        "wokring": "working",
        "blongs": "belongs",
        "belngs": "belongs",
        "haz": "has",
        "haas": "has",
        "diffrence": "difference",
        "differnce": "difference",
    }

    def __init__(self):
        self._lock = threading.RLock()
        self._cached_dataset_hash: str = ""
        self._dataset_name: str = ""
        
        # Vocabulary maps
        self._raw_values: Set[str] = set()
        self._columns: List[str] = []
        self._column_type_map: Dict[str, str] = {}
        
        # Multi-representation indexes
        self._exact_val_map: Dict[str, Tuple[str, str]] = {}     # lower -> (original_val, col)
        self._compact_val_map: Dict[str, Tuple[str, str]] = {}   # compact -> (original_val, col)
        self._punct_free_map: Dict[str, Tuple[str, str]] = {}    # punct_free -> (original_val, col)
        
        # Column indexes
        self._exact_col_map: Dict[str, str] = {}                 # lower -> col
        self._compact_col_map: Dict[str, str] = {}               # compact -> col
        
        # Phonetic Index
        self._phonetic_index = PhoneticIndex()
        
        # Token to value mapping for multi-word phrases
        self._token_to_values: Dict[str, Set[str]] = {}

    def build_vocabulary(self, df: pd.DataFrame, dataset_name: str = "", dataset_hash: str = "") -> None:
        """Extract vocabulary from DataFrame columns and distinct values, building multi-index structures."""
        with self._lock:
            if dataset_hash and dataset_hash == self._cached_dataset_hash and self._cached_dataset_hash != "":
                logger.debug(f"DatasetSpellChecker: Vocabulary already built for hash {dataset_hash[:8]}")
                return

            self._clear()
            self._cached_dataset_hash = dataset_hash
            self._dataset_name = dataset_name

            if df is None or df.empty:
                return

            self._columns = [c for c in df.columns if not c.startswith("_internal_")]

            # 1. Index Column Names
            for col in self._columns:
                col_clean = str(col).strip()
                col_lower = col_clean.lower()
                self._exact_col_map[col_lower] = col_clean
                compact = re.sub(r"[^a-z0-9]", "", col_lower)
                if compact:
                    self._compact_col_map[compact] = col_clean
                
                # Tag semantic type
                col_l = col_lower
                if "state" in col_l:
                    self._column_type_map[col_clean] = "STATE"
                elif "capital" in col_l:
                    self._column_type_map[col_clean] = "CAPITAL"
                elif "city" in col_l or "location" in col_l:
                    self._column_type_map[col_clean] = "CITY"
                elif any(k in col_l for k in ["name", "employee", "customer", "person", "student"]):
                    self._column_type_map[col_clean] = "PERSON"
                elif any(k in col_l for k in ["product", "item", "sku"]):
                    self._column_type_map[col_clean] = "PRODUCT"
                elif any(k in col_l for k in ["dept", "department", "division"]):
                    self._column_type_map[col_clean] = "DEPARTMENT"
                elif any(k in col_l for k in ["salary", "price", "amount", "cost", "score", "rating"]):
                    self._column_type_map[col_clean] = "NUMERIC"
                else:
                    self._column_type_map[col_clean] = "GENERAL"

            # 2. Index Column Values
            for col in self._columns:
                col_clean = str(col).strip()
                unique_vals = df[col].dropna().unique()
                for val in unique_vals:
                    val_str = str(val).strip()
                    if not val_str or val_str.lower() in {"nan", "none", "null", "n/a", "-"}:
                        continue

                    # Don't index purely numeric values as words
                    if val_str.replace(".", "", 1).isdigit():
                        continue

                    self._raw_values.add(val_str)
                    val_lower = val_str.lower()
                    val_norm = unicodedata.normalize("NFKD", val_lower)

                    # Exact lower mapping
                    if val_lower not in self._exact_val_map:
                        self._exact_val_map[val_lower] = (val_str, col_clean)

                    # Compact mapping (no spaces, hyphens, underscores, slashes)
                    compact = re.sub(r"[^a-z0-9]", "", val_lower)
                    if compact and compact not in self._compact_val_map:
                        self._compact_val_map[compact] = (val_str, col_clean)

                    # Punctuation-free mapping
                    punct_free = re.sub(r"[\-_/.,;']", " ", val_lower).strip()
                    punct_free = re.sub(r"\s+", " ", punct_free)
                    if punct_free and punct_free not in self._punct_free_map:
                        self._punct_free_map[punct_free] = (val_str, col_clean)

                    # Phonetic index
                    self._phonetic_index.add_term(val_str)

                    # Token index for multi-word values
                    tokens = [t for t in re.split(r"[\s\-_/]+", val_lower) if len(t) >= 3]
                    for tok in tokens:
                        self._token_to_values.setdefault(tok, set()).add(val_str)

                    # Dynamic Alias Binding from EntityAliasResolver
                    try:
                        from app.dataset.alias_resolver import entity_alias_resolver
                        STOPWORD_ALIASES = {"as", "or", "in", "is", "at", "to", "on", "do", "no", "so", "it", "an", "if", "by"}
                        # Check state aliases
                        for alias_k, canon_s in entity_alias_resolver.state_codes.items():
                            if canon_s.lower() == val_lower:
                                ak_lower = alias_k.lower()
                                if ak_lower in STOPWORD_ALIASES:
                                    continue
                                ak_compact = re.sub(r"[^a-z0-9]", "", ak_lower)
                                if ak_lower not in self._exact_val_map:
                                    self._exact_val_map[ak_lower] = (val_str, col_clean)
                                if ak_compact and ak_compact not in self._compact_val_map:
                                    self._compact_val_map[ak_compact] = (val_str, col_clean)
                                self._phonetic_index.add_term(alias_k)
                        # Check city & historical aliases
                        for alias_k, canon_c in list(entity_alias_resolver.city_codes.items()) + list(entity_alias_resolver.historical_names.items()):
                            if canon_c.lower() == val_lower:
                                ak_lower = alias_k.lower()
                                if ak_lower in STOPWORD_ALIASES:
                                    continue
                                ak_compact = re.sub(r"[^a-z0-9]", "", ak_lower)
                                if ak_lower not in self._exact_val_map:
                                    self._exact_val_map[ak_lower] = (val_str, col_clean)
                                if ak_compact and ak_compact not in self._compact_val_map:
                                    self._compact_val_map[ak_compact] = (val_str, col_clean)
                                self._phonetic_index.add_term(alias_k)
                    except Exception:
                        pass

            logger.info(
                f"DatasetSpellChecker: Built vocabulary for '{dataset_name}' with {len(self._raw_values)} unique values across {len(self._columns)} columns."
            )

    def add_dataset_vocabulary(self, df: pd.DataFrame, dataset_name: str = "") -> None:
        """Incrementally index columns and values from an additional dataset without clearing existing vocabulary."""
        with self._lock:
            if df is None or df.empty:
                return

            cols = [str(c).strip() for c in df.columns if not str(c).startswith("_internal_")]
            for col in cols:
                col_clean = str(col).strip()
                col_lower = col_clean.lower()
                if col_lower not in self._exact_col_map:
                    self._exact_col_map[col_lower] = col_clean
                    compact = re.sub(r"[^a-z0-9]", "", col_lower)
                    if compact:
                        self._compact_col_map[compact] = col_clean

                unique_vals = df[col].dropna().unique()
                for val in unique_vals:
                    val_str = str(val).strip()
                    if not val_str or val_str.lower() in {"nan", "none", "null", "n/a", "-"}:
                        continue
                    if val_str.replace(".", "", 1).isdigit() and len(val_str) > 6:
                        continue

                    val_lower = val_str.lower()
                    self._raw_values.add(val_str)
                    if val_lower not in self._exact_val_map:
                        self._exact_val_map[val_lower] = (val_str, col_clean)
                    compact = re.sub(r"[^a-z0-9]", "", val_lower)
                    if compact and compact not in self._compact_val_map:
                        self._compact_val_map[compact] = (val_str, col_clean)

                    punct_free = re.sub(r"[\-_/.,;']", " ", val_lower).strip()
                    punct_free = re.sub(r"\s+", " ", punct_free)
                    if punct_free and punct_free not in self._punct_free_map:
                        self._punct_free_map[punct_free] = (val_str, col_clean)

                    self._phonetic_index.add_term(val_str)

                    tokens = [t for t in re.split(r"[\s\-_/]+", val_lower) if len(t) >= 3]
                    for tok in tokens:
                        self._token_to_values.setdefault(tok, set()).add(val_str)

    def _clear(self) -> None:
        """Reset internal vocabulary maps."""
        self._raw_values.clear()
        self._columns.clear()
        self._column_type_map.clear()
        self._exact_val_map.clear()
        self._compact_val_map.clear()
        self._punct_free_map.clear()
        self._exact_col_map.clear()
        self._compact_col_map.clear()
        self._phonetic_index = PhoneticIndex()
        self._token_to_values.clear()

    def resolve_candidate(
        self,
        raw_token: str,
        expected_type: Optional[str] = None,
        column_context: Optional[str] = None
    ) -> Optional[SpellingCorrection]:
        """Resolve misspelled token or phrase using 14-stage priority order against dataset vocabulary."""
        if not raw_token or len(raw_token.strip()) < 2:
            return None

        clean_tok = raw_token.strip()

        # 0. Check ID pattern (e.g. EMP-1001, emp1001, SKU-ABC-20)
        id_corr = self._check_id_pattern(clean_tok, column_context)
        if id_corr:
            return id_corr

        tok_lower = clean_tok.lower()
        tok_norm = unicodedata.normalize("NFKD", tok_lower)
        tok_compact = re.sub(r"[^a-z0-9]", "", tok_norm)
        tok_punct_free = re.sub(r"[\-_/.,;']", " ", tok_norm).strip()
        tok_punct_free = re.sub(r"\s+", " ", tok_punct_free)

        # -------------------------------------------------------------
        # Stage 1: Exact Match
        # -------------------------------------------------------------
        for orig_val in self._raw_values:
            if clean_tok == orig_val:
                col = self._get_column_for_value(orig_val, column_context)
                return SpellingCorrection(clean_tok, orig_val, col, 1.0, ["exact_match"])

        # -------------------------------------------------------------
        # Stage 2: Case-Insensitive Match
        # -------------------------------------------------------------
        if tok_lower in self._exact_val_map:
            val, col = self._exact_val_map[tok_lower]
            return SpellingCorrection(clean_tok, val, col, 1.0, ["case_insensitive"])

        # -------------------------------------------------------------
        # Stage 3: Unicode Normalization Match
        # -------------------------------------------------------------
        for k, (v, col) in self._exact_val_map.items():
            if tok_norm == unicodedata.normalize("NFKD", k):
                return SpellingCorrection(clean_tok, v, col, 0.99, ["unicode_normalized"])

        # -------------------------------------------------------------
        # Stage 4 & 5: Punctuation & Whitespace Normalization (e.g. West-Bengal, West  Bengal)
        # -------------------------------------------------------------
        if tok_punct_free in self._punct_free_map:
            val, col = self._punct_free_map[tok_punct_free]
            return SpellingCorrection(clean_tok, val, col, 0.98, ["whitespace_punctuation_normalized"])

        # -------------------------------------------------------------
        # Stage 6: Compact Form (Missing spaces, e.g. "westbengal" -> "West Bengal")
        # -------------------------------------------------------------
        if tok_compact in self._compact_val_map:
            val, col = self._compact_val_map[tok_compact]
            return SpellingCorrection(clean_tok, val, col, 0.97, ["compact_form"])

        # -------------------------------------------------------------
        # Stage 7 & 8: Known Alias & Abbreviation Resolution (AP, TG, WB, HYD, BLR)
        # -------------------------------------------------------------
        cols_to_search = [column_context] if column_context else self._columns
        alias_res = entity_alias_resolver.resolve_alias(clean_tok, expected_type=expected_type, available_columns=cols_to_search)
        if alias_res and alias_res.get("canonical_name"):
            canon_name = alias_res["canonical_name"]
            canon_lower = canon_name.lower()
            if canon_lower in self._exact_val_map:
                val, col = self._exact_val_map[canon_lower]
                return SpellingCorrection(clean_tok, val, col, alias_res.get("confidence", 0.96), ["alias_resolution"])
            # Even if exact value is not in exact_val_map, return the resolved canon_name
            return SpellingCorrection(clean_tok, canon_name, column_context, alias_res.get("confidence", 0.95), ["alias_resolution"])

        # -------------------------------------------------------------
        # Stage 9: Character-level Edit Distance (Levenshtein: deletion, insertion, substitution, transposition)
        # -------------------------------------------------------------
        edit_dist_res = self._match_by_edit_distance(clean_tok, tok_lower, tok_compact, expected_type, column_context)
        if edit_dist_res and edit_dist_res.confidence >= 0.88:
            return edit_dist_res

        # -------------------------------------------------------------
        # Stage 10: RapidFuzz Multi-Metric Matching
        # -------------------------------------------------------------
        rf_res = self._match_by_rapidfuzz(clean_tok, tok_lower, expected_type, column_context)
        if rf_res and rf_res.confidence >= 0.85:
            return rf_res

        # -------------------------------------------------------------
        # Stage 11: Phonetic Matching (Double Metaphone, Soundex, Indian Phonetics)
        # -------------------------------------------------------------
        phon_res = self._match_by_phonetics(clean_tok, expected_type, column_context)
        if phon_res and phon_res.confidence >= 0.80:
            return phon_res

        # If we have an edit distance result with moderate confidence
        if edit_dist_res and edit_dist_res.confidence >= 0.75:
            return edit_dist_res

        if rf_res and rf_res.confidence >= 0.75:
            return rf_res

        return None

    def resolve_column(
        self,
        candidate_text: str,
        available_columns: Optional[List[str]] = None
    ) -> Optional[ColumnMatch]:
        """Resolve misspelled column name using exact, compact, and RapidFuzz matching."""
        if not candidate_text:
            return None

        clean_text = candidate_text.strip().lower()
        cols = available_columns if available_columns is not None else self._columns

        exact_map = {c.lower(): c for c in cols}
        compact_map = {re.sub(r"[^a-z0-9]", "", c.lower()): c for c in cols}

        # 1. Exact match
        if clean_text in exact_map:
            col = exact_map[clean_text]
            return ColumnMatch(col, 1.0, "exact_column_match")

        # 2. Compact match (e.g. "performancescore" -> "Performance Score")
        compact = re.sub(r"[^a-z0-9]", "", clean_text)
        if compact in compact_map:
            col = compact_map[compact]
            return ColumnMatch(col, 0.97, "compact_column_match")

        # 3. Common query typos (e.g. "captial" -> "capital", "salry" -> "salary")
        if clean_text in self.QUERY_WORD_CORRECTIONS:
            corrected = self.QUERY_WORD_CORRECTIONS[clean_text]
            if corrected in exact_map:
                col = exact_map[corrected]
                return ColumnMatch(col, 0.96, "query_column_typo")

        # 4. RapidFuzz against available columns (use whole-string and token-sort ratio, not substring WRatio)
        best_col = None
        best_score = 0.0
        for col in cols:
            col_l = col.lower()
            r1 = fuzz.ratio(clean_text, col_l)
            r2 = fuzz.token_sort_ratio(clean_text, col_l)
            words_cand = clean_text.split()
            words_col = col_l.split()
            if len(words_cand) != len(words_col):
                r_effective = r1
            else:
                word_ratios = [fuzz.ratio(w1, w2) for w1, w2 in zip(sorted(words_cand), sorted(words_col))]
                if any(wr < 55.0 for wr in word_ratios):
                    r_effective = r1
                else:
                    r_effective = max(r1, r2)
            if r_effective > best_score:
                best_score = r_effective
                best_col = col

        if best_score >= 75.0 and best_col:
            conf = round(best_score / 100.0, 2)
            return ColumnMatch(best_col, conf, "rapidfuzz_column_match")

        return None

    def correct_full_question(self, question: str) -> FullQuestionCorrection:
        """Normalize typos across the entire question sentence while protecting dataset IDs."""
        if not question or not question.strip():
            return FullQuestionCorrection(question, [])

        tokens = question.split()
        corrections: List[SpellingCorrection] = []
        new_tokens: List[str] = []

        i = 0
        n = len(tokens)
        while i < n:
            raw_w = tokens[i]
            # Strip trailing punctuation for inspection but preserve it for reattachment
            clean_w = re.sub(r"^[^\w]+", "", raw_w)
            lead_punct = raw_w[:len(raw_w) - len(clean_w)]
            tail_punct = ""
            m_tail = re.search(r"[^\w]+$", clean_w)
            if m_tail:
                tail_punct = m_tail.group(0)
                clean_w = clean_w[:-len(tail_punct)]

            # Check possessive 's or s'
            is_possessive = False
            if clean_w.endswith("'s") or clean_w.endswith("’s"):
                clean_w = clean_w[:-2]
                is_possessive = True

            # 1. Check ID protection (e.g. EMP-1001, emp1001, CUS-1005)
            if self._is_id_token(clean_w):
                new_tokens.append(raw_w)
                i += 1
                continue

            # 2. Check 2-word phrase first (e.g. "west bengal", "performence score", "joinig date")
            STOPWORDS_START = {
                "of", "in", "at", "for", "from", "to", "by", "with", "the", "a", "an", "is", "are", "has", "have",
                "and", "or", "does", "which", "what", "how", "who", "paid", "earns", "earning", "makes", "making",
                "highest", "lowest", "more", "less", "top", "best", "most", "least"
            }
            handled_phrase = False
            if i + 1 < n and clean_w.lower() not in STOPWORDS_START:
                next_raw = tokens[i + 1]
                next_clean = re.sub(r"[^\w]", "", next_raw)
                two_word = f"{clean_w} {next_clean}".lower()
                
                # Check column match (only if matched column is multi-word)
                col_corr = self.resolve_column(two_word)
                if col_corr and len(str(col_corr).split()) > 1 and col_corr.confidence >= 0.85:
                    new_tokens.append(col_corr.corrected_value)
                    corrections.append(col_corr.to_spelling_correction(two_word))
                    i += 2
                    continue

                # Check entity match (only if target entity is multi-word)
                ent_corr = self.resolve_candidate(two_word)
                if ent_corr and len(ent_corr.corrected_value.split()) > 1 and ent_corr.confidence >= 0.85:
                    rep = ent_corr.corrected_value + ("'s" if is_possessive else "") + tail_punct
                    new_tokens.append(rep)
                    corrections.append(ent_corr)
                    i += 2
                    continue

            # 3. Check single-word column typo
            col_corr = self.resolve_column(clean_w.lower())
            if col_corr and col_corr.confidence >= 0.88 and clean_w.lower() != col_corr.corrected_value.lower():
                rep = lead_punct + col_corr.corrected_value.lower() + tail_punct
                new_tokens.append(rep)
                corrections.append(col_corr.to_spelling_correction(clean_w))
                i += 1
                continue

            # 4. Check single-word query typos (e.g. higest -> highest, salry -> salary)
            w_lower = clean_w.lower()
            if w_lower in self.QUERY_WORD_CORRECTIONS:
                fixed_w = self.QUERY_WORD_CORRECTIONS[w_lower]
                rep = lead_punct + fixed_w + tail_punct
                new_tokens.append(rep)
                corrections.append(SpellingCorrection(clean_w, fixed_w, None, 0.99, ["query_word_dictionary"]))
                i += 1
                continue

            # 5. Check single-word dataset entity typo (e.g. telengana -> Telangana, sikim -> Sikkim, hydrabad -> Hyderabad)
            ent_corr = self.resolve_candidate(clean_w)
            if ent_corr and ent_corr.confidence >= 0.80 and clean_w.lower() != ent_corr.corrected_value.lower():
                COMMON_ENGLISH_STOPWORDS = {
                    "in", "is", "of", "to", "for", "the", "a", "an", "has", "have", "had", "are", "do", "does", "did",
                    "as", "or", "at", "by", "on", "from", "with", "which", "what", "how", "who", "where", "when", "why",
                    "many", "much", "more", "most", "less", "least", "count", "total", "each", "every", "all", "some", "any",
                    "tell", "me", "show", "give", "find", "list", "search", "get", "print", "about", "can", "you", "please",
                    "bro", "dude", "now", "fast", "yo", "hey", "bhai", "yaar", "zara", "whats", "gimme", "kya", "hai",
                    "batao", "bataiye", "chahiye", "rajdhani", "pls", "having"
                }
                if clean_w.lower() not in COMMON_ENGLISH_STOPWORDS or clean_w.isupper():
                    rep = lead_punct + ent_corr.corrected_value + ("'s" if is_possessive else "") + tail_punct
                    new_tokens.append(rep)
                    corrections.append(ent_corr)
                    if i + 1 < n:
                        next_raw = tokens[i + 1]
                        next_clean = re.sub(r"[^\w]", "", next_raw).lower()
                        if next_clean and next_clean in ent_corr.corrected_value.lower().split():
                            i += 2
                            continue
                    i += 1
                    continue

            # No correction needed
            new_tokens.append(raw_w)
            i += 1

        corrected_q = " ".join(new_tokens)
        return FullQuestionCorrection(corrected_q, corrections)

    # -------------------------------------------------------------------------
    # MATCHING STAGES IMPLEMENTATION
    # -------------------------------------------------------------------------
    def _match_by_edit_distance(
        self,
        orig_token: str,
        tok_lower: str,
        tok_compact: str,
        expected_type: Optional[str],
        column_context: Optional[str]
    ) -> Optional[SpellingCorrection]:
        """Calculate Damerau-Levenshtein edit distance (deletions, insertions, substitutions, transpositions)."""
        best_candidate: Optional[str] = None
        best_col: Optional[str] = None
        min_dist = 999
        max_ratio = 0.0

        candidates = self._filter_values_by_type(expected_type, column_context)

        for val_str, col_name in candidates:
            val_l = val_str.lower()
            val_compact = re.sub(r"[^a-z0-9]", "", val_l)

            # Test standard distance and compact distance
            d1 = distance.DamerauLevenshtein.distance(tok_lower, val_l)
            d2 = distance.DamerauLevenshtein.distance(tok_compact, val_compact)
            dist = min(d1, d2)

            # Max allowed edit distance based on word length
            # len <= 4: max dist 1
            # len 5..8: max dist 2
            # len >= 9: max dist 3
            allowed_dist = 1 if len(tok_lower) <= 4 else (2 if len(tok_lower) <= 8 else 3)
            if dist <= allowed_dist and dist < min_dist:
                sim = 1.0 - (dist / max(len(tok_lower), len(val_l)))
                min_dist = dist
                max_ratio = sim
                best_candidate = val_str
                best_col = col_name

        if best_candidate and max_ratio >= 0.75:
            conf = round(max(0.75, max_ratio), 2)
            return SpellingCorrection(orig_token, best_candidate, best_col, conf, ["edit_distance"])

        return None

    def _match_by_rapidfuzz(
        self,
        orig_token: str,
        tok_lower: str,
        expected_type: Optional[str],
        column_context: Optional[str]
    ) -> Optional[SpellingCorrection]:
        """Apply RapidFuzz ratio, WRatio, and token sort metrics."""
        candidates = self._filter_values_by_type(expected_type, column_context)
        scored: List[Tuple[float, str, str]] = []

        for val_str, col_name in candidates:
            val_l = val_str.lower()
            r_ratio = fuzz.ratio(tok_lower, val_l)
            r_wratio = fuzz.WRatio(tok_lower, val_l)
            r_sort = fuzz.token_sort_ratio(tok_lower, val_l)
            r_set = fuzz.token_set_ratio(tok_lower, val_l)

            # Weighted composite score
            composite = (r_ratio * 0.40) + (r_wratio * 0.30) + (r_sort * 0.15) + (r_set * 0.15)
            if composite >= 75.0:
                scored.append((composite, val_str, col_name))

        if not scored:
            return None

        scored.sort(key=lambda x: x[0], reverse=True)
        top_score, top_val, top_col = scored[0]

        # Check ambiguity among top candidates
        is_ambig = False
        cand_list = [top_val]
        if len(scored) > 1 and abs(top_score - scored[1][0]) < 3.0:
            is_ambig = True
            cand_list.append(scored[1][1])

        conf = round(top_score / 100.0, 2)
        return SpellingCorrection(
            orig_token, top_val, top_col, conf, ["rapidfuzz_matching"], is_ambiguous=is_ambig, candidates=cand_list
        )

    def _match_by_phonetics(
        self,
        orig_token: str,
        expected_type: Optional[str],
        column_context: Optional[str]
    ) -> Optional[SpellingCorrection]:
        """Evaluate Double Metaphone, Soundex, and Indian phonetic similarity."""
        # 1. Retrieve candidate terms from phonetic index
        cand_terms = self._phonetic_index.get_candidates(orig_token)
        if not cand_terms:
            cand_terms = set(self._raw_values)

        scored: List[Tuple[float, str, str]] = []
        for val_str in cand_terms:
            col_name = self._get_column_for_value(val_str, column_context)
            # Filter by expected type if specified
            if expected_type and self._column_type_map.get(col_name) != expected_type:
                continue

            ps = phonetic_match_score(orig_token, val_str)
            if ps >= 0.78:
                scored.append((ps, val_str, col_name))

        if not scored:
            return None

        scored.sort(key=lambda x: x[0], reverse=True)
        top_score, top_val, top_col = scored[0]

        is_ambig = False
        cand_list = [top_val]
        if len(scored) > 1 and abs(top_score - scored[1][0]) < 0.03:
            is_ambig = True
            cand_list.append(scored[1][1])

        return SpellingCorrection(
            orig_token, top_val, top_col, round(top_score, 2), ["phonetic_matching"], is_ambiguous=is_ambig, candidates=cand_list
        )

    # -------------------------------------------------------------------------
    # HELPERS
    # -------------------------------------------------------------------------
    def _is_id_token(self, token: str) -> bool:
        """Check if token is an alphanumeric code or ID (e.g. EMP-1001, emp1001, SKU-90)."""
        tok = token.strip()
        # Must contain digits
        if not any(ch.isdigit() for ch in tok):
            return False
        # Matches patterns like EMP-1001, SKU-ABC-20, CUS102
        if re.match(r"^[A-Za-z]+[-_]?\d+$", tok, re.IGNORECASE) or re.match(r"^[A-Za-z]+[-_][A-Za-z0-9]+[-_]\d+$", tok):
            return True
        return False

    def _check_id_pattern(self, token: str, column_context: Optional[str]) -> Optional[SpellingCorrection]:
        """Normalize format of ID tokens without using standard spell checker."""
        if not self._is_id_token(token):
            return None

        clean_t = token.upper().replace("-", "").replace("_", "")
        # Search dataset values in ID columns
        for orig_val in self._raw_values:
            clean_val = str(orig_val).upper().replace("-", "").replace("_", "")
            if clean_t == clean_val:
                col = self._get_column_for_value(orig_val, column_context)
                return SpellingCorrection(token, orig_val, col, 1.0, ["id_format_normalization"])
        return None

    def _filter_values_by_type(
        self,
        expected_type: Optional[str],
        column_context: Optional[str]
    ) -> List[Tuple[str, str]]:
        """Filter dataset values according to expected semantic type or column context."""
        results: List[Tuple[str, str]] = []
        for val_str in self._raw_values:
            col = self._get_column_for_value(val_str, column_context)
            if column_context and col != column_context:
                continue
            if expected_type:
                col_type = self._column_type_map.get(col, "GENERAL")
                if col_type != expected_type and col_type != "GENERAL":
                    continue
            results.append((val_str, col))
        return results

    def _get_column_for_value(self, value: str, preferred_col: Optional[str] = None) -> str:
        """Find the column containing a specific unique value."""
        val_l = value.strip().lower()
        if val_l in self._exact_val_map:
            return self._exact_val_map[val_l][1]
        return preferred_col or (self._columns[0] if self._columns else "entity")


dataset_spell_checker = DatasetSpellChecker()
