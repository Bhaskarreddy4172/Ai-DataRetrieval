"""Semantic column mapper with Universal Ontology, dynamic discovery, and confidence scoring."""

import difflib
from typing import Any, Dict, List, Optional, Set, Tuple
from app.utils.logger import logger

try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False


# Section 4: Universal Semantic Vocabulary / Ontology
UNIVERSAL_ONTOLOGY: Dict[str, Dict[str, Any]] = {
    "PERSON": {
        "synonyms": ["person", "individual", "people", "someone", "somebody", "guy", "dude", "candidate", "human"],
        "category": "entity"
    },
    "EMPLOYEE": {
        "synonyms": ["employee", "worker", "staff", "colleague", "member", "associate", "team member", "personnel", "emp", "hire"],
        "category": "entity"
    },
    "CUSTOMER": {
        "synonyms": ["customer", "client", "buyer", "shopper", "consumer", "patron", "subscriber", "account", "cust"],
        "category": "entity"
    },
    "STUDENT": {
        "synonyms": ["student", "pupil", "scholar", "learner", "enrollee"],
        "category": "entity"
    },
    "PRODUCT": {
        "synonyms": ["product", "item", "good", "merchandise", "sku", "article", "unit", "gadget", "device"],
        "category": "entity"
    },
    "ORDER": {
        "synonyms": ["order", "transaction", "purchase", "booking", "invoice", "sale"],
        "category": "entity"
    },
    "LOCATION": {
        "synonyms": ["location", "place", "site", "venue", "area", "region", "zone", "address", "branch"],
        "category": "dimension"
    },
    "CITY": {
        "synonyms": ["city", "town", "metro", "municipality", "blr", "hyd", "mum", "del", "chennai", "bengaluru", "hyderabad", "mumbai", "delhi"],
        "category": "dimension"
    },
    "COUNTRY": {
        "synonyms": ["country", "nation", "state", "territory", "province"],
        "category": "dimension"
    },
    "DEPARTMENT": {
        "synonyms": ["department", "dept", "division", "team", "unit", "sector", "branch", "group", "practice"],
        "category": "dimension"
    },
    "TEAM": {
        "synonyms": ["team", "squad", "pod", "crew", "group"],
        "category": "dimension"
    },
    "CATEGORY": {
        "synonyms": ["category", "type", "kind", "class", "classification", "segment", "genre"],
        "category": "dimension"
    },
    "DESIGNATION": {
        "synonyms": ["designation", "role", "title", "position", "job", "job title", "post", "level"],
        "category": "dimension"
    },
    "PRICE": {
        "synonyms": ["price", "cost", "charge", "rate", "fee", "pricing", "tag", "retail price", "mrp"],
        "category": "metric"
    },
    "COST": {
        "synonyms": ["cost", "expense", "expenditure", "spend", "spending", "outlay"],
        "category": "metric"
    },
    "SALARY": {
        "synonyms": ["salary", "pay", "paid", "earns", "earning", "making", "gets", "getting", "draws", "drawing", "paycheck", "compensation", "wage", "wages", "earnings", "income", "stipend", "remuneration", "cash", "bank", "package", "ctc"],
        "category": "metric"
    },
    "REVENUE": {
        "synonyms": ["revenue", "turnover", "sales", "gross revenue", "income", "receipts"],
        "category": "metric"
    },
    "PROFIT": {
        "synonyms": ["profit", "margin", "net income", "gain", "earnings"],
        "category": "metric"
    },
    "QUANTITY": {
        "synonyms": ["quantity", "stock", "count", "inventory", "units", "volume", "amount"],
        "category": "metric"
    },
    "COUNT": {
        "synonyms": ["count", "number", "total", "volume", "quantity", "heads", "headcount"],
        "category": "metric"
    },
    "DATE": {
        "synonyms": ["date", "day", "time", "timestamp", "joined", "joining", "created", "updated", "expiry", "start", "end", "when"],
        "category": "temporal"
    },
    "TIME": {
        "synonyms": ["time", "hour", "duration", "tenure", "timing"],
        "category": "temporal"
    },
    "RATING": {
        "synonyms": ["rating", "stars", "review", "grade", "feedback", "score"],
        "category": "metric"
    },
    "SCORE": {
        "synonyms": ["score", "performance", "rating", "points", "marks", "result", "grade", "index", "kpi"],
        "category": "metric"
    },
    "STATUS": {
        "synonyms": ["status", "state", "condition", "stage", "phase", "active", "pending", "completed"],
        "category": "dimension"
    },
    "ID": {
        "synonyms": ["id", "identifier", "code", "number", "key", "ref", "reference", "sku", "emp id", "cust id"],
        "category": "identifier"
    },
    "NAME": {
        "synonyms": ["name", "title", "full name", "first name", "last name", "label", "called"],
        "category": "identifier"
    },
    "EMAIL": {
        "synonyms": ["email", "mail", "email address", "contact email"],
        "category": "contact"
    },
    "PHONE": {
        "synonyms": ["phone", "mobile", "contact", "cell", "telephone", "number"],
        "category": "contact"
    },
    "AGE": {
        "synonyms": ["age", "years old", "dob", "birthdate"],
        "category": "metric"
    },
    "EXPERIENCE": {
        "synonyms": ["experience", "tenure", "years of experience", "seniority"],
        "category": "metric"
    },
    "TOTAL": {
        "synonyms": ["total", "sum", "aggregate", "overall", "gross", "entire"],
        "category": "metric"
    },
    "AMOUNT": {
        "synonyms": ["amount", "value", "sum", "bill", "total amount", "charge"],
        "category": "metric"
    },
    "COMPENSATION": {
        "synonyms": ["compensation", "remuneration", "package", "ctc", "stipend", "pay", "salary", "bonus"],
        "category": "metric"
    },
    "DISCOUNT": {
        "synonyms": ["discount", "rebate", "concession", "markdown", "deduction", "offer"],
        "category": "metric"
    },
    "TAX": {
        "synonyms": ["tax", "gst", "vat", "duty", "tariff", "cess", "withholding"],
        "category": "metric"
    },
    "ADDRESS": {
        "synonyms": ["address", "street", "road", "residence", "domicile", "premise", "locality"],
        "category": "dimension"
    },
    "STATE": {
        "synonyms": ["state", "province", "region", "territory", "sub-division", "st"],
        "category": "dimension"
    },
    "CAPITAL": {
        "synonyms": ["capital", "capital city", "headquarters", "seat", "principal city"],
        "category": "dimension"
    },
    "CURRENCY": {
        "synonyms": ["currency", "fx", "forex", "denomination", "usd", "inr", "eur", "gbp"],
        "category": "dimension"
    },
    "UNIT": {
        "synonyms": ["unit", "uom", "measurement", "scale", "dimension", "metric unit"],
        "category": "dimension"
    }
}


class SemanticColumnMapper:
    """Resolves natural language terms to dataset columns with universal ontology and confidence guards."""

    def __init__(self):
        self.ontology = UNIVERSAL_ONTOLOGY

    def discover_column_semantics(self, column_name: str, sample_values: Optional[List[Any]] = None) -> Dict[str, Any]:
        """Dynamic dataset semantic discovery (Section 5 & 16): infer concepts, aliases, and confidence."""
        col_clean = column_name.strip().lower()
        col_words = set(col_clean.replace("_", " ").replace("-", " ").split())

        matched_concepts = []
        confidence = 0.5

        for concept_name, info in self.ontology.items():
            synonyms = info["synonyms"]
            # Exact concept name match
            if col_clean == concept_name.lower() or concept_name.lower() in col_words:
                matched_concepts.append(concept_name.lower())
                confidence = max(confidence, 0.98)
            # Synonym matches
            for syn in synonyms:
                if syn == col_clean:
                    matched_concepts.append(concept_name.lower())
                    confidence = max(confidence, 0.95)
                elif syn in col_words or col_clean in syn:
                    matched_concepts.append(concept_name.lower())
                    confidence = max(confidence, 0.85)

        # Value inspection if provided
        if sample_values:
            non_null = [str(v) for v in sample_values if v is not None and str(v).strip()][:5]
            if any("@" in str(v) for v in non_null):
                matched_concepts.append("email")
                confidence = max(confidence, 0.99)
            elif any(str(v).startswith(("EMP-", "CUST-", "ORD-", "PROD-", "ID-")) for v in non_null):
                matched_concepts.append("id")
                confidence = max(confidence, 0.99)

        # Unique concepts
        matched_concepts = list(dict.fromkeys(matched_concepts))
        if not matched_concepts:
            matched_concepts = ["dimension" if sample_values and isinstance(sample_values[0], str) else "metric"]

        aliases = list(set([col_clean, col_clean.replace(" ", "_"), col_clean.replace("_", " ")] + matched_concepts))

        return {
            "column": column_name,
            "semantic_concepts": matched_concepts,
            "confidence": round(confidence, 2),
            "aliases": aliases
        }

    def map_semantic_concept(
        self,
        concept: str,
        available_columns: List[str],
        schema_dict: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Maps a high-level semantic concept (e.g. 'compensation', 'spending', 'place') to actual column."""
        concept_clean = concept.strip().lower()

        # Check ontology synonyms for this concept
        candidates: List[str] = [concept_clean]
        for c_key, info in self.ontology.items():
            if c_key.lower() == concept_clean or concept_clean in info["synonyms"]:
                candidates.extend(info["synonyms"])

        candidates = list(dict.fromkeys(candidates))

        # Score against available columns
        scored: List[Tuple[float, str]] = []
        for col in available_columns:
            col_l = col.lower()
            col_words = col_l.replace("_", " ").replace("-", " ").split()
            for cand in candidates:
                if cand == col_l or cand in col_words:
                    scored.append((0.95, col))
                elif cand in col_l or col_l in cand:
                    scored.append((0.85, col))

        if scored:
            scored.sort(key=lambda x: x[0], reverse=True)
            return {"column": scored[0][1], "confidence": scored[0][0], "candidates": [s[1] for s in scored]}

        # Fallback to map_column
        return self.map_column(concept, available_columns, schema_dict)

    def map_column(
        self,
        candidate_term: str,
        available_columns: List[str],
        schema_dict: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Map user phrase to best matching column, with confidence and ambiguity checks."""
        if not candidate_term or not available_columns:
            return {"column": None, "confidence": 0.0, "is_ambiguous": False, "candidates": []}

        term_clean = candidate_term.strip().lower()

        # Build search variants (handling plurals like categories -> category, cities -> city)
        variants = [term_clean]
        if term_clean.endswith("ies") and len(term_clean) > 3:
            variants.append(term_clean[:-3] + "y")
        elif term_clean.endswith("s") and not term_clean.endswith("ss") and len(term_clean) > 2:
            variants.append(term_clean[:-1])

        # 1. Exact Column Match
        for col in available_columns:
            for v in variants:
                if v == col.lower():
                    return {"column": col, "confidence": 1.0, "is_ambiguous": False, "candidates": [col]}

        # 2. Check Schema Aliases & Universal Ontology
        scored_matches: List[Tuple[float, str]] = []
        if schema_dict and "columns" in schema_dict:
            cols_info = schema_dict["columns"]
            if isinstance(cols_info, dict):
                col_items = cols_info.items()
            elif isinstance(cols_info, list):
                col_items = [(c["name"], c) for c in cols_info if isinstance(c, dict) and "name" in c]
            else:
                col_items = []

            for col_name, info in col_items:
                aliases = [a.lower() for a in info.get("aliases", [])]
                for v in variants:
                    if v in aliases:
                        scored_matches.append((0.98, col_name))
                    elif any(v in a or a in v for a in aliases):
                        scored_matches.append((0.90, col_name))

        # Check Universal Ontology Synonyms
        for concept_key, info in self.ontology.items():
            syns = info["synonyms"]
            if any(v in syns for v in variants):
                for col in available_columns:
                    col_l = col.lower()
                    if concept_key.lower() in col_l or any(s in col_l for s in syns if len(s) > 3):
                        scored_matches.append((0.94, col))

        # 3. Substring in Column Name
        for col in available_columns:
            col_lower = col.lower()
            for v in variants:
                if v in col_lower or col_lower in v:
                    scored_matches.append((0.92, col))

        # 4. Fuzzy Similarity on Column Names and Words (RapidFuzz if available, else difflib)
        for col in available_columns:
            for v in variants:
                if HAS_RAPIDFUZZ:
                    ratio = fuzz.ratio(v, col.lower()) / 100.0
                else:
                    ratio = difflib.SequenceMatcher(None, v, col.lower()).ratio()

                if ratio >= 0.65:
                    scored_matches.append((round(ratio, 2), col))
                for w in col.lower().split():
                    if HAS_RAPIDFUZZ:
                        w_ratio = fuzz.ratio(v, w) / 100.0
                    else:
                        w_ratio = difflib.SequenceMatcher(None, v, w).ratio()
                    if w_ratio >= 0.75:
                        scored_matches.append((round(w_ratio * 0.95, 2), col))

        if not scored_matches:
            return {"column": None, "confidence": 0.0, "is_ambiguous": False, "candidates": []}

        # Deduplicate and sort by confidence descending
        unique_candidates: Dict[str, float] = {}
        for score, col in scored_matches:
            if col not in unique_candidates or score > unique_candidates[col]:
                unique_candidates[col] = score

        sorted_candidates = sorted(unique_candidates.items(), key=lambda x: x[1], reverse=True)
        best_col, best_score = sorted_candidates[0]

        # 5. Check Ambiguity: If second best candidate is within 0.06 of top score
        is_ambiguous = False
        competing = [c[0] for c in sorted_candidates if c[1] >= best_score - 0.06]
        if len(competing) > 1 and best_score < 0.98:
            is_ambiguous = True

        return {
            "column": best_col if not is_ambiguous else None,
            "confidence": best_score,
            "is_ambiguous": is_ambiguous,
            "candidates": competing,
            "top_candidates": [c[0] for c in sorted_candidates[:3]]
        }


semantic_column_mapper = SemanticColumnMapper()

