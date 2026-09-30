"""Universal Entity Alias & Abbreviation Resolution Engine.

Resolves:
- Indian State and UT abbreviations and short codes (AP, TG, TS, WB, MH, DL, etc.)
- Major city airport/IATA codes and colloquial abbreviations (HYD, BLR, BOM, CCU, MAA, DEL, PNQ, etc.)
- Historical and alternate names (Bangalore <-> Bengaluru, Bombay <-> Mumbai, Madras <-> Chennai, Calcutta <-> Kolkata, etc.)
- Department and business abbreviations (HR, IT, QA, R&D, ENG, MKTG, etc.)
- Global country and currency codes (USA, UK, UAE, USD, INR, EUR, etc.)
- Dynamic dataset-generated acronyms (e.g. West Bengal -> WB, Human Resources -> HR)
"""

import re
from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd
from app.utils.logger import logger


# 1. State & Union Territory Canonical Registry with Standard Codes
INDIAN_STATE_CODES: Dict[str, str] = {
    "ap": "Andhra Pradesh",
    "ar": "Arunachal Pradesh",
    "as": "Assam",
    "br": "Bihar",
    "cg": "Chhattisgarh",
    "ct": "Chhattisgarh",
    "ga": "Goa",
    "gj": "Gujarat",
    "hr": "Haryana",
    "hp": "Himachal Pradesh",
    "jh": "Jharkhand",
    "ka": "Karnataka",
    "kar": "Karnataka",
    "kl": "Kerala",
    "ker": "Kerala",
    "mp": "Madhya Pradesh",
    "mh": "Maharashtra",
    "maha": "Maharashtra",
    "mn": "Manipur",
    "ml": "Meghalaya",
    "meg": "Meghalaya",
    "mz": "Mizoram",
    "miz": "Mizoram",
    "nl": "Nagaland",
    "nag": "Nagaland",
    "od": "Odisha",
    "or": "Odisha",
    "pb": "Punjab",
    "pun": "Punjab",
    "rj": "Rajasthan",
    "raj": "Rajasthan",
    "sk": "Sikkim",
    "sik": "Sikkim",
    "tn": "Tamil Nadu",
    "ts": "Telangana",
    "tg": "Telangana",
    "tel": "Telangana",
    "tr": "Tripura",
    "tri": "Tripura",
    "up": "Uttar Pradesh",
    "uk": "Uttarakhand",
    "ut": "Uttarakhand",
    "ua": "Uttarakhand",
    "wb": "West Bengal",
    # Colloquial / Common Typo Mappings
    "andra": "Andhra Pradesh",
    "andra pradesh": "Andhra Pradesh",
    "andrapradesh": "Andhra Pradesh",
    "andhrapradesh": "Andhra Pradesh",
    "arunachal": "Arunachal Pradesh",
    "arunachalpradesh": "Arunachal Pradesh",
    "asom": "Assam",
    "bihaar": "Bihar",
    "chattisgarh": "Chhattisgarh",
    "chatisgarh": "Chhattisgarh",
    "gujrat": "Gujarat",
    "himachal": "Himachal Pradesh",
    "himachalpradesh": "Himachal Pradesh",
    "jharkand": "Jharkhand",
    "karnatak": "Karnataka",
    "karnatka": "Karnataka",
    "madhyapradesh": "Madhya Pradesh",
    "rajastan": "Rajasthan",
    "sikim": "Sikkim",
    "cikkim": "Sikkim",
    "sikkimm": "Sikkim",
    "tamilnadu": "Tamil Nadu",
    "telengana": "Telangana",
    "uttarpradesh": "Uttar Pradesh",
    "uttaranchal": "Uttarakhand",
    "westbengal": "West Bengal",
    "west bengol": "West Bengal",
    # Union Territories
    "an": "Andaman and Nicobar Islands",
    "ch": "Chandigarh",
    "chd": "Chandigarh",
    "dh": "Dadra and Nagar Haveli and Daman and Diu",
    "dd": "Dadra and Nagar Haveli and Daman and Diu",
    "dnh": "Dadra and Nagar Haveli and Daman and Diu",
    "dl": "Delhi",
    "nct": "Delhi",
    "jk": "Jammu and Kashmir",
    "jnk": "Jammu and Kashmir",
    "la": "Ladakh",
    "ld": "Lakshadweep",
    "py": "Puducherry",
    "pondy": "Puducherry",
    "pondicherry": "Puducherry",
}

# 2. Major Cities, Airport/IATA Codes, and Colloquial Names
CITY_AIRPORT_CODES: Dict[str, str] = {
    "hyd": "Hyderabad",
    "blr": "Bengaluru",
    "bng": "Bengaluru",
    "bom": "Mumbai",
    "mum": "Mumbai",
    "ccu": "Kolkata",
    "cal": "Kolkata",
    "maa": "Chennai",
    "che": "Chennai",
    "del": "Delhi",
    "ndls": "Delhi",
    "pnq": "Pune",
    "amd": "Ahmedabad",
    "jai": "Jaipur",
    "lko": "Lucknow",
    "bbsr": "Bhubaneswar",
    "bbi": "Bhubaneswar",
    "cok": "Kochi",
    "trv": "Thiruvananthapuram",
    "goi": "Goa",
    "pat": "Patna",
    "rpr": "Raipur",
    "rnc": "Ranchi",
    "gau": "Guwahati",
    "ixc": "Chandigarh",
    "idr": "Indore",
    "nag": "Nagpur",
    "vtz": "Visakhapatnam",
    "vga": "Vijayawada",
    "trz": "Tiruchirappalli",
    "cjb": "Coimbatore",
    "ixb": "Bagdogra",
    "ixr": "Ranchi",
    "ixa": "Agartala",
    "ixs": "Silchar",
    "imf": "Imphal",
    "dmu": "Dimapur",
    "shl": "Shillong",
    "vns": "Varanasi",
    "atq": "Amritsar",
    "sxr": "Srinagar",
    "ixe": "Mangaluru",
    "bdq": "Vadodara",
    "svx": "Surat",
    "stv": "Surat",
    "bpl": "Bhopal",
    "ddn": "Dehradun",
    "gnr": "Gandhinagar",
    "pnj": "Panaji",
}

# 3. Alternate / Historical / Colloquial Names bidirectional map
HISTORICAL_AND_ALTERNATE_NAMES: Dict[str, str] = {
    "bangalore": "Bengaluru",
    "bengaluru": "Bengaluru",
    "bombay": "Mumbai",
    "mumbai": "Mumbai",
    "madras": "Chennai",
    "chennai": "Chennai",
    "calcutta": "Kolkata",
    "kolkata": "Kolkata",
    "amaravathi": "Amaravati",
    "amarawati": "Amaravati",
    "panjim": "Panaji",
    "dehradoon": "Dehradun",
    "bhubaneshwar": "Bhubaneswar",
    "cochin": "Kochi",
    "kochi": "Kochi",
    "trivandrum": "Thiruvananthapuram",
    "thiruvananthapuram": "Thiruvananthapuram",
    "poona": "Pune",
    "pune": "Pune",
    "baroda": "Vadodara",
    "vadodara": "Vadodara",
    "waltair": "Visakhapatnam",
    "vizag": "Visakhapatnam",
    "visakhapatnam": "Visakhapatnam",
    "vishakhapatnam": "Visakhapatnam",
    "simla": "Shimla",
    "shimla": "Shimla",
    "pondicherry": "Puducherry",
    "puducherry": "Puducherry",
    "orissa": "Odisha",
    "odisha": "Odisha",
    "mysore": "Mysuru",
    "mysuru": "Mysuru",
    "mangalore": "Mangaluru",
    "mangaluru": "Mangaluru",
    "belgaum": "Belagavi",
    "belagavi": "Belagavi",
    "calicut": "Kozhikode",
    "kozhikode": "Kozhikode",
    "gurgaon": "Gurugram",
    "gurugram": "Gurugram",
    "allahabad": "Prayagraj",
    "prayagraj": "Prayagraj",
    "benaras": "Varanasi",
    "banaras": "Varanasi",
    "kashi": "Varanasi",
    "varanasi": "Varanasi",
    "secunderabad": "Secunderabad",
    "cyberabad": "Hyderabad",
    "gauhati": "Guwahati",
    "guwahati": "Guwahati",
}

# 4. Department and Business Abbreviations
DEPARTMENT_CODES: Dict[str, str] = {
    "hr": "Human Resources",
    "it": "Information Technology",
    "qa": "Quality Assurance",
    "qc": "Quality Control",
    "r&d": "Research & Development",
    "rnd": "Research & Development",
    "eng": "Engineering",
    "engg": "Engineering",
    "mktg": "Marketing",
    "mkt": "Marketing",
    "fin": "Finance",
    "ops": "Operations",
    "admin": "Administration",
    "adm": "Administration",
    "sales": "Sales",
    "dev": "Development",
    "mgmt": "Management",
    "support": "Support",
    "cs": "Customer Support",
    "legal": "Legal",
    "pr": "Public Relations",
}

# 5. Global Country & Currency Codes
GLOBAL_CODES: Dict[str, str] = {
    "usa": "United States",
    "us": "United States",
    "uk": "United Kingdom",
    "uae": "United Arab Emirates",
    "in": "India",
    "ind": "India",
    "ca": "Canada",
    "can": "Canada",
    "au": "Australia",
    "aus": "Australia",
    "sg": "Singapore",
    "de": "Germany",
    "fr": "France",
    "jp": "Japan",
    "jpn": "Japan",
    "cn": "China",
    "chn": "China",
    "usd": "US Dollar",
    "inr": "Indian Rupee",
    "eur": "Euro",
    "gbp": "British Pound",
    "jpy": "Japanese Yen",
    "aud": "Australian Dollar",
    "cad": "Canadian Dollar",
    "sgd": "Singapore Dollar",
    "aed": "UAE Dirham",
}

# Mapping of Indian Cities to their parent State
CITY_TO_STATE_MAP: Dict[str, str] = {
    "hyderabad": "Telangana",
    "secunderabad": "Telangana",
    "warangal": "Telangana",
    "karimnagar": "Telangana",
    "nizamabad": "Telangana",
    "khammam": "Telangana",
    "bengaluru": "Karnataka",
    "bangalore": "Karnataka",
    "mysuru": "Karnataka",
    "mysore": "Karnataka",
    "mangaluru": "Karnataka",
    "mangalore": "Karnataka",
    "hubballi": "Karnataka",
    "hubli": "Karnataka",
    "belagavi": "Karnataka",
    "belgaum": "Karnataka",
    "mumbai": "Maharashtra",
    "bombay": "Maharashtra",
    "pune": "Maharashtra",
    "poona": "Maharashtra",
    "nagpur": "Maharashtra",
    "nashik": "Maharashtra",
    "aurangabad": "Maharashtra",
    "thane": "Maharashtra",
    "navi mumbai": "Maharashtra",
    "kolkata": "West Bengal",
    "calcutta": "West Bengal",
    "howrah": "West Bengal",
    "durgapur": "West Bengal",
    "asansol": "West Bengal",
    "siliguri": "West Bengal",
    "chennai": "Tamil Nadu",
    "madras": "Tamil Nadu",
    "coimbatore": "Tamil Nadu",
    "madurai": "Tamil Nadu",
    "tiruchirappalli": "Tamil Nadu",
    "salem": "Tamil Nadu",
    "delhi": "Delhi",
    "new delhi": "Delhi",
    "noida": "Uttar Pradesh",
    "greater noida": "Uttar Pradesh",
    "ghaziabad": "Uttar Pradesh",
    "lucknow": "Uttar Pradesh",
    "kanpur": "Uttar Pradesh",
    "varanasi": "Uttar Pradesh",
    "prayagraj": "Uttar Pradesh",
    "allahabad": "Uttar Pradesh",
    "agra": "Uttar Pradesh",
    "gurugram": "Haryana",
    "gurgaon": "Haryana",
    "faridabad": "Haryana",
    "panipat": "Haryana",
    "chandigarh": "Chandigarh",
    "jaipur": "Rajasthan",
    "jodhpur": "Rajasthan",
    "udaipur": "Rajasthan",
    "kota": "Rajasthan",
    "ahmedabad": "Gujarat",
    "surat": "Gujarat",
    "vadodara": "Gujarat",
    "baroda": "Gujarat",
    "rajkot": "Gujarat",
    "gandhinagar": "Gujarat",
    "bhopal": "Madhya Pradesh",
    "indore": "Madhya Pradesh",
    "gwalior": "Madhya Pradesh",
    "jabalpur": "Madhya Pradesh",
    "patna": "Bihar",
    "gaya": "Bihar",
    "bhagalpur": "Bihar",
    "muzaffarpur": "Bihar",
    "ranchi": "Jharkhand",
    "jamshedpur": "Jharkhand",
    "dhanbad": "Jharkhand",
    "raipur": "Chhattisgarh",
    "bilaspur": "Chhattisgarh",
    "bhubaneswar": "Odisha",
    "cuttack": "Odisha",
    "rourkela": "Odisha",
    "puri": "Odisha",
    "thiruvananthapuram": "Kerala",
    "trivandrum": "Kerala",
    "kochi": "Kerala",
    "cochin": "Kerala",
    "kozhikode": "Kerala",
    "calicut": "Kerala",
    "visakhapatnam": "Andhra Pradesh",
    "vizag": "Andhra Pradesh",
    "waltair": "Andhra Pradesh",
    "vijayawada": "Andhra Pradesh",
    "guntur": "Andhra Pradesh",
    "tirupati": "Andhra Pradesh",
    "amaravati": "Andhra Pradesh",
    "guwahati": "Assam",
    "dispur": "Assam",
    "panaji": "Goa",
    "vasco": "Goa",
    "shimla": "Himachal Pradesh",
    "dharamshala": "Himachal Pradesh",
    "dehradun": "Uttarakhand",
    "amritsar": "Punjab",
    "ludhiana": "Punjab",
    "jalandhar": "Punjab",
    "srinagar": "Jammu and Kashmir",
    "jammu": "Jammu and Kashmir",
    "gangtok": "Sikkim",
    "shillong": "Meghalaya",
    "aizawl": "Mizoram",
    "kohima": "Nagaland",
    "imphal": "Manipur",
    "agartala": "Tripura",
    "itanagar": "Arunachal Pradesh",
    "port blair": "Andaman and Nicobar Islands",
    "leh": "Ladakh",
    "puducherry": "Puducherry",
    "pondicherry": "Puducherry",
}

# State Capitals Map (Canonical State -> Capital)
STATE_TO_CAPITAL_MAP: Dict[str, str] = {
    "Andhra Pradesh": "Amaravati",
    "Arunachal Pradesh": "Itanagar",
    "Assam": "Dispur",
    "Bihar": "Patna",
    "Chhattisgarh": "Raipur",
    "Goa": "Panaji",
    "Gujarat": "Gandhinagar",
    "Haryana": "Chandigarh",
    "Himachal Pradesh": "Shimla",
    "Jharkhand": "Ranchi",
    "Karnataka": "Bengaluru",
    "Kerala": "Thiruvananthapuram",
    "Madhya Pradesh": "Bhopal",
    "Maharashtra": "Mumbai",
    "Manipur": "Imphal",
    "Meghalaya": "Shillong",
    "Mizoram": "Aizawl",
    "Nagaland": "Kohima",
    "Odisha": "Bhubaneswar",
    "Punjab": "Chandigarh",
    "Rajasthan": "Jaipur",
    "Sikkim": "Gangtok",
    "Tamil Nadu": "Chennai",
    "Telangana": "Hyderabad",
    "Tripura": "Agartala",
    "Uttar Pradesh": "Lucknow",
    "Uttarakhand": "Dehradun",
    "West Bengal": "Kolkata",
    "Andaman and Nicobar Islands": "Port Blair",
    "Chandigarh": "Chandigarh",
    "Dadra and Nagar Haveli and Daman and Diu": "Daman",
    "Delhi": "New Delhi",
    "Jammu and Kashmir": "Srinagar",
    "Ladakh": "Leh",
    "Lakshadweep": "Kavaratti",
    "Puducherry": "Puducherry",
}


class EntityAliasResolver:
    """Universal Entity Alias & Abbreviation Resolution Engine."""

    def __init__(self):
        self.state_codes = INDIAN_STATE_CODES
        self.city_codes = CITY_AIRPORT_CODES
        self.historical_names = HISTORICAL_AND_ALTERNATE_NAMES
        self.department_codes = DEPARTMENT_CODES
        self.global_codes = GLOBAL_CODES
        self.city_to_state = CITY_TO_STATE_MAP
        self.state_to_capital = STATE_TO_CAPITAL_MAP
        # Dynamic acronyms generated per dataset: {acronym_lower: (canonical_value, column_name)}
        self.dynamic_acronyms: Dict[str, List[Tuple[str, str]]] = {}
        # Dynamic full text index: {token_lower: [(canonical_value, column_name)]}
        self.dynamic_token_map: Dict[str, List[Tuple[str, str]]] = {}

    def index_dataset(self, df: pd.DataFrame) -> None:
        """Dynamically generate acronyms, concatenated forms, and token indices from dataset."""
        self.dynamic_acronyms.clear()
        self.dynamic_token_map.clear()

        if df is None or df.empty:
            return

        for col in df.columns:
            series = df[col].dropna().unique()
            for raw_val in series:
                val_str = str(raw_val).strip()
                if not val_str or len(val_str) < 2:
                    continue

                val_lower = val_str.lower()
                tokens = re.findall(r"[A-Za-z0-9]+", val_str)

                # 1. Index full normalized value
                if val_lower not in self.dynamic_token_map:
                    self.dynamic_token_map[val_lower] = []
                self.dynamic_token_map[val_lower].append((val_str, col))

                # 2. Generate Acronym for multi-word values (e.g. "West Bengal" -> "WB", "Human Resources" -> "HR")
                if len(tokens) >= 2:
                    # First letters of words
                    acronym = "".join(t[0].lower() for t in tokens if t)
                    if len(acronym) >= 2 and len(acronym) <= 6:
                        if acronym not in self.dynamic_acronyms:
                            self.dynamic_acronyms[acronym] = []
                        if (val_str, col) not in self.dynamic_acronyms[acronym]:
                            self.dynamic_acronyms[acronym].append((val_str, col))

                    # Dotted acronym (e.g. "w.b.")
                    dotted = ".".join(t[0].lower() for t in tokens if t)
                    if dotted not in self.dynamic_acronyms:
                        self.dynamic_acronyms[dotted] = []
                    if (val_str, col) not in self.dynamic_acronyms[dotted]:
                        self.dynamic_acronyms[dotted].append((val_str, col))

                    # Concatenated form (e.g. "westbengal")
                    concat = "".join(tokens).lower()
                    if concat not in self.dynamic_token_map:
                        self.dynamic_token_map[concat] = []
                    if (val_str, col) not in self.dynamic_token_map[concat]:
                        self.dynamic_token_map[concat].append((val_str, col))

        logger.info(f"AliasResolver dynamically indexed {len(self.dynamic_acronyms)} acronyms from dataset.")

    def resolve_alias(
        self,
        candidate_token: str,
        expected_type: Optional[str] = None,
        available_columns: Optional[List[str]] = None,
        context_text: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Resolve a candidate token (abbreviation, code, alternate name, typo) to canonical entity with context scoring."""
        if not candidate_token:
            return None

        cand_raw = candidate_token.strip()
        cand_clean = cand_raw.lower().replace(".", "").replace("'", "").replace("-", "")
        cand_lower = cand_raw.lower()

        # Guard: Ambiguous common 2-letter English words should not be treated as codes
        # unless uppercase in query or explicit context provided
        COMMON_ENGLISH_WORDS = {"in", "is", "at", "to", "or", "as", "an", "on", "if", "do", "by", "no", "so"}
        if cand_clean in COMMON_ENGLISH_WORDS:
            # If lowercase and no matching context, reject
            if not cand_raw.isupper() and not expected_type and not available_columns and not context_text:
                return None
            # "in" can only be India (Country) if explicitly Country/Global or uppercase
            if cand_clean == "in" and expected_type in ["CITY", "STATE", "DEPARTMENT", "LOCATION", "GEOGRAPHY"]:
                return None
            # "or" / "as" can only be State if explicitly State/Location or uppercase
            if cand_clean in ["or", "as"] and expected_type not in ["STATE", "LOCATION", "GEOGRAPHY"]:
                return None

        candidates: List[Dict[str, Any]] = []

        # 1. State Candidate
        if cand_clean in self.state_codes:
            canon_state = self.state_codes[cand_clean]
            candidates.append({
                "canonical_name": canon_state,
                "matched_alias": cand_raw,
                "entity_type": "STATE",
                "confidence": 0.98,
                "target_column_hints": ["State", "Region", "Province", "Location", "State Name"],
                "capital": self.state_to_capital.get(canon_state),
            })

        # 2. City Candidate (Airport/IATA)
        if cand_clean in self.city_codes:
            canon_city = self.city_codes[cand_clean]
            parent_st = self.city_to_state.get(canon_city.lower())
            candidates.append({
                "canonical_name": canon_city,
                "matched_alias": cand_raw,
                "entity_type": "CITY",
                "confidence": 0.98,
                "target_column_hints": ["City", "Location", "Place", "Town", "Metro", "Branch"],
                "parent_state": parent_st,
            })

        # 3. Alternate / Historical Name Candidate
        if cand_lower in self.historical_names or cand_clean in self.historical_names:
            key = cand_lower if cand_lower in self.historical_names else cand_clean
            canon_hist = self.historical_names[key]
            parent_st = self.city_to_state.get(canon_hist.lower())
            candidates.append({
                "canonical_name": canon_hist,
                "matched_alias": cand_raw,
                "entity_type": "CITY" if parent_st else "LOCATION",
                "confidence": 0.99,
                "target_column_hints": ["City", "Location", "Place", "State"],
                "parent_state": parent_st,
            })

        # 4. Department Candidate
        if cand_clean in self.department_codes or cand_lower in self.department_codes:
            key = cand_lower if cand_lower in self.department_codes else cand_clean
            canon_dept = self.department_codes[key]
            candidates.append({
                "canonical_name": canon_dept,
                "matched_alias": cand_raw,
                "entity_type": "DEPARTMENT",
                "confidence": 0.98,
                "target_column_hints": ["Department", "Dept", "Division", "Team", "Unit"],
            })

        # 5. Global Country / Currency Candidate
        if cand_clean in self.global_codes:
            canon_glob = self.global_codes[cand_clean]
            is_currency = "Dollar" in canon_glob or "Rupee" in canon_glob or "Euro" in canon_glob or "Pound" in canon_glob or "Yen" in canon_glob or "Dirham" in canon_glob
            candidates.append({
                "canonical_name": canon_glob,
                "matched_alias": cand_raw,
                "entity_type": "CURRENCY" if is_currency else "COUNTRY",
                "confidence": 0.97,
                "target_column_hints": ["Currency", "Price", "Salary"] if is_currency else ["Country", "Nation", "Location"],
            })

        # 6. Dynamic Dataset Acronym Candidate
        if cand_clean in self.dynamic_acronyms or cand_lower in self.dynamic_acronyms:
            key = cand_lower if cand_lower in self.dynamic_acronyms else cand_clean
            for best_val, best_col in self.dynamic_acronyms[key]:
                candidates.append({
                    "canonical_name": best_val,
                    "matched_alias": cand_raw,
                    "entity_type": "DYNAMIC_DATASET_ENTITY",
                    "confidence": 0.95,
                    "target_column_hints": [best_col],
                })

        # 7. Dynamic Token / Concatenated Candidate
        if cand_clean in self.dynamic_token_map or cand_lower in self.dynamic_token_map:
            key = cand_lower if cand_lower in self.dynamic_token_map else cand_clean
            for best_val, best_col in self.dynamic_token_map[key]:
                candidates.append({
                    "canonical_name": best_val,
                    "matched_alias": cand_raw,
                    "entity_type": "DYNAMIC_DATASET_ENTITY",
                    "confidence": 0.96,
                    "target_column_hints": [best_col],
                })

        if not candidates:
            return None

        # Filter by expected_type if given (strict filtering)
        if expected_type:
            exp_upper = expected_type.upper()
            filtered = [c for c in candidates if c["entity_type"] == exp_upper or (exp_upper in ["LOCATION", "GEOGRAPHY"] and c["entity_type"] in ["STATE", "CITY"])]
            if not filtered:
                return None
            candidates = filtered

        # Context scoring function
        cols_lower = [col.lower() for col in (available_columns or [])]
        ctx_lower = (context_text or "").lower()

        def score_candidate(cand: Dict[str, Any]) -> float:
            score = cand.get("confidence", 0.9)
            hints = [h.lower() for h in cand.get("target_column_hints", [])]

            # 1. Column Match Boost (highest priority: candidate matches available dataset column)
            if cols_lower:
                if any(h in cols_lower or any(h in c for c in cols_lower) for h in hints):
                    score += 1.0
                else:
                    score -= 0.2

            # 2. Context Text Match Boost
            if ctx_lower:
                if any(h in ctx_lower for h in hints):
                    score += 0.5
                if cand["entity_type"].lower() in ctx_lower:
                    score += 0.3

            # 3. Disambiguation defaults for ambiguous codes when no context provided:
            # e.g., 'UK' defaults to United Kingdom unless State context is present
            if cand_clean == "uk":
                if cand["entity_type"] == "COUNTRY" and not any("state" in c for c in cols_lower):
                    score += 0.05
            # e.g., 'HR' defaults to Department if dataset has Department column
            elif cand_clean == "hr":
                if any("dept" in c or "department" in c for c in cols_lower):
                    if cand["entity_type"] == "DEPARTMENT":
                        score += 0.5

            return score

        # Sort candidates by calculated score descending
        candidates.sort(key=score_candidate, reverse=True)
        return candidates[0]

    def resolve_capital_of(self, state_or_code: str) -> Optional[Dict[str, str]]:
        """Resolve capital of state, shortcut, or alias (e.g. 'TG' -> Hyderabad, 'WB' -> Kolkata)."""
        if not state_or_code:
            return None

        clean_term = state_or_code.strip()
        alias_res = self.resolve_alias(clean_term, expected_type="STATE")
        state_name = alias_res["canonical_name"] if alias_res else clean_term

        # Look up in capital map
        for canon_state, capital in self.state_to_capital.items():
            if canon_state.lower() == state_name.lower():
                return {
                    "state": canon_state,
                    "capital": capital,
                    "input": state_or_code,
                    "matched_alias": alias_res["matched_alias"] if alias_res else state_or_code
                }

        return None

    def resolve_state_of(self, city_or_code: str) -> Optional[Dict[str, str]]:
        """Resolve parent state of city, airport code, or alias (e.g. 'HYD' -> Telangana, 'BLR' -> Karnataka)."""
        if not city_or_code:
            return None

        clean_term = city_or_code.strip()
        alias_res = self.resolve_alias(clean_term, expected_type="CITY")
        city_name = alias_res["canonical_name"] if alias_res else clean_term

        city_lower = city_name.lower()
        if city_lower in self.city_to_state:
            state_name = self.city_to_state[city_lower]
            return {
                "city": city_name,
                "state": state_name,
                "input": city_or_code,
                "matched_alias": alias_res["matched_alias"] if alias_res else city_or_code
            }

        return None


# Global singleton instance
entity_alias_resolver = EntityAliasResolver()
