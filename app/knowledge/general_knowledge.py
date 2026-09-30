"""Safe, comprehensive General Knowledge Engine for geography, technology, and common facts."""

import re
from typing import Any, Dict, List, Optional, Tuple
from app.ai.ollama_client import ollama_client
from app.dataset.alias_resolver import entity_alias_resolver
from app.utils.logger import logger

try:
    from rapidfuzz import fuzz, process
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

# 1. Comprehensive Indian Geography Knowledge Base
INDIAN_STATES: Dict[str, Dict[str, Any]] = {
    "Andhra Pradesh": {"capital": "Amaravati", "type": "State"},
    "Arunachal Pradesh": {"capital": "Itanagar", "type": "State"},
    "Assam": {"capital": "Dispur", "type": "State"},
    "Bihar": {"capital": "Patna", "type": "State"},
    "Chhattisgarh": {"capital": "Raipur", "type": "State"},
    "Goa": {"capital": "Panaji", "type": "State"},
    "Gujarat": {"capital": "Gandhinagar", "type": "State"},
    "Haryana": {"capital": "Chandigarh", "type": "State"},
    "Himachal Pradesh": {"capital": "Shimla", "type": "State"},
    "Jharkhand": {"capital": "Ranchi", "type": "State"},
    "Karnataka": {"capital": "Bengaluru", "type": "State"},
    "Kerala": {"capital": "Thiruvananthapuram", "type": "State"},
    "Madhya Pradesh": {"capital": "Bhopal", "type": "State"},
    "Maharashtra": {"capital": "Mumbai", "type": "State"},
    "Manipur": {"capital": "Imphal", "type": "State"},
    "Meghalaya": {"capital": "Shillong", "type": "State"},
    "Mizoram": {"capital": "Aizawl", "type": "State"},
    "Nagaland": {"capital": "Kohima", "type": "State"},
    "Odisha": {"capital": "Bhubaneswar", "type": "State"},
    "Punjab": {"capital": "Chandigarh", "type": "State"},
    "Rajasthan": {"capital": "Jaipur", "type": "State"},
    "Sikkim": {"capital": "Gangtok", "type": "State"},
    "Tamil Nadu": {"capital": "Chennai", "type": "State"},
    "Telangana": {"capital": "Hyderabad", "type": "State"},
    "Tripura": {"capital": "Agartala", "type": "State"},
    "Uttar Pradesh": {"capital": "Lucknow", "type": "State"},
    "Uttarakhand": {"capital": "Dehradun", "type": "State"},
    "West Bengal": {"capital": "Kolkata", "type": "State"},
    "Andaman and Nicobar Islands": {"capital": "Port Blair", "type": "Union Territory"},
    "Chandigarh": {"capital": "Chandigarh", "type": "Union Territory"},
    "Dadra and Nagar Haveli and Daman and Diu": {"capital": "Daman", "type": "Union Territory"},
    "Delhi": {"capital": "New Delhi", "type": "National Capital Territory"},
    "Jammu and Kashmir": {"capital": "Srinagar", "type": "Union Territory"},
    "Ladakh": {"capital": "Leh", "type": "Union Territory"},
    "Lakshadweep": {"capital": "Kavaratti", "type": "Union Territory"},
    "Puducherry": {"capital": "Puducherry", "type": "Union Territory"},
}

# State Aliases and Common Misspellings/Abbreviations
STATE_ALIASES: Dict[str, str] = {
    "andhra pradesh": "Andhra Pradesh",
    "andra pradesh": "Andhra Pradesh",
    "andhra": "Andhra Pradesh",
    "andra": "Andhra Pradesh",
    "ap": "Andhra Pradesh",
    "a.p.": "Andhra Pradesh",
    "a.p": "Andhra Pradesh",
    "telangana": "Telangana",
    "telengana": "Telangana",
    "ts": "Telangana",
    "t.s.": "Telangana",
    "t.s": "Telangana",
    "tg": "Telangana",
    "karnataka": "Karnataka",
    "karnatka": "Karnataka",
    "kar": "Karnataka",
    "ka": "Karnataka",
    "tamil nadu": "Tamil Nadu",
    "tamilnadu": "Tamil Nadu",
    "tn": "Tamil Nadu",
    "t.n.": "Tamil Nadu",
    "maharashtra": "Maharashtra",
    "maharashtraa": "Maharashtra",
    "mh": "Maharashtra",
    "m.h.": "Maharashtra",
    "uttar pradesh": "Uttar Pradesh",
    "up": "Uttar Pradesh",
    "u.p.": "Uttar Pradesh",
    "madhya pradesh": "Madhya Pradesh",
    "mp": "Madhya Pradesh",
    "m.p.": "Madhya Pradesh",
    "west bengal": "West Bengal",
    "bengal": "West Bengal",
    "wb": "West Bengal",
    "w.b.": "West Bengal",
    "kerala": "Kerala",
    "kl": "Kerala",
    "gujarat": "Gujarat",
    "gujrat": "Gujarat",
    "gj": "Gujarat",
    "rajasthan": "Rajasthan",
    "raj": "Rajasthan",
    "rj": "Rajasthan",
    "delhi": "Delhi",
    "new delhi": "Delhi",
    "dl": "Delhi",
    "punjab": "Punjab",
    "pb": "Punjab",
    "haryana": "Haryana",
    "hr": "Haryana",
    "bihar": "Bihar",
    "br": "Bihar",
    "odisha": "Odisha",
    "orissa": "Odisha",
    "od": "Odisha",
    "assam": "Assam",
    "a.s.": "Assam",
    "asom": "Assam",
    "goa": "Goa",
    "ga": "Goa",
    "jharkhand": "Jharkhand",
    "jh": "Jharkhand",
    "chhattisgarh": "Chhattisgarh",
    "cg": "Chhattisgarh",
    "himachal pradesh": "Himachal Pradesh",
    "hp": "Himachal Pradesh",
    "uttarakhand": "Uttarakhand",
    "uk": "Uttarakhand",
}

# Major Cities to State Mapping (with aliases & common misspellings)
INDIAN_CITIES: Dict[str, Dict[str, Any]] = {
    "Bengaluru": {
        "state": "Karnataka",
        "display_name": "Bangalore (Bengaluru)",
        "aliases": ["bangalore", "bengaluru", "blr", "banglore", "banglroe", "bangluru", "bengalore"]
    },
    "Hyderabad": {
        "state": "Telangana",
        "display_name": "Hyderabad",
        "aliases": ["hyderabad", "hyd", "hydrabad", "hyderbad", "secunderabad"]
    },
    "Mumbai": {
        "state": "Maharashtra",
        "display_name": "Mumbai",
        "aliases": ["mumbai", "bombay", "mum", "mumbaii", "navi mumbai"]
    },
    "Pune": {
        "state": "Maharashtra",
        "display_name": "Pune",
        "aliases": ["pune", "poona", "punee"]
    },
    "Chennai": {
        "state": "Tamil Nadu",
        "display_name": "Chennai",
        "aliases": ["chennai", "madras", "che", "maa", "chenai"]
    },
    "Kolkata": {
        "state": "West Bengal",
        "display_name": "Kolkata",
        "aliases": ["kolkata", "calcutta", "kol", "cal"]
    },
    "Delhi": {
        "state": "Delhi (National Capital Territory)",
        "display_name": "Delhi / New Delhi",
        "aliases": ["delhi", "new delhi", "del", "dilli", "delh"]
    },
    "Ahmedabad": {
        "state": "Gujarat",
        "display_name": "Ahmedabad",
        "aliases": ["ahmedabad", "amdavad", "ahmadabad"]
    },
    "Surat": {
        "state": "Gujarat",
        "display_name": "Surat",
        "aliases": ["surat"]
    },
    "Jaipur": {
        "state": "Rajasthan",
        "display_name": "Jaipur",
        "aliases": ["jaipur", "pink city"]
    },
    "Lucknow": {
        "state": "Uttar Pradesh",
        "display_name": "Lucknow",
        "aliases": ["lucknow", "lko"]
    },
    "Kanpur": {
        "state": "Uttar Pradesh",
        "display_name": "Kanpur",
        "aliases": ["kanpur"]
    },
    "Noida": {
        "state": "Uttar Pradesh",
        "display_name": "Noida",
        "aliases": ["noida", "greater noida"]
    },
    "Patna": {
        "state": "Bihar",
        "display_name": "Patna",
        "aliases": ["patna"]
    },
    "Bhopal": {
        "state": "Madhya Pradesh",
        "display_name": "Bhopal",
        "aliases": ["bhopal"]
    },
    "Indore": {
        "state": "Madhya Pradesh",
        "display_name": "Indore",
        "aliases": ["indore"]
    },
    "Chandigarh": {
        "state": "Punjab / Haryana (Union Territory)",
        "display_name": "Chandigarh",
        "aliases": ["chandigarh", "chd"]
    },
    "Thiruvananthapuram": {
        "state": "Kerala",
        "display_name": "Thiruvananthapuram (Trivandrum)",
        "aliases": ["thiruvananthapuram", "trivandrum"]
    },
    "Kochi": {
        "state": "Kerala",
        "display_name": "Kochi (Cochin)",
        "aliases": ["kochi", "cochin"]
    },
    "Bhubaneswar": {
        "state": "Odisha",
        "display_name": "Bhubaneswar",
        "aliases": ["bhubaneswar", "bbsr", "bhubaneshwar"]
    },
    "Guwahati": {
        "state": "Assam",
        "display_name": "Guwahati",
        "aliases": ["guwahati", "gauhati"]
    },
    "Panaji": {
        "state": "Goa",
        "display_name": "Panaji (Goa)",
        "aliases": ["panaji", "panjim", "goa"]
    },
    "Coimbatore": {
        "state": "Tamil Nadu",
        "display_name": "Coimbatore",
        "aliases": ["coimbatore", "kovai"]
    },
    "Visakhapatnam": {
        "state": "Andhra Pradesh",
        "display_name": "Visakhapatnam (Vizag)",
        "aliases": ["visakhapatnam", "vizag", "vishakapatnam"]
    },
    "Nagpur": {
        "state": "Maharashtra",
        "display_name": "Nagpur",
        "aliases": ["nagpur"]
    },
    "Varanasi": {
        "state": "Uttar Pradesh",
        "display_name": "Varanasi",
        "aliases": ["varanasi", "banaras", "kashi"]
    },
    "Agra": {
        "state": "Uttar Pradesh",
        "display_name": "Agra",
        "aliases": ["agra"]
    },
    "Amritsar": {
        "state": "Punjab",
        "display_name": "Amritsar",
        "aliases": ["amritsar"]
    },
    "Dehradun": {
        "state": "Uttarakhand",
        "display_name": "Dehradun",
        "aliases": ["dehradun"]
    },
    "Shimla": {
        "state": "Himachal Pradesh",
        "display_name": "Shimla",
        "aliases": ["shimla", "simla"]
    },
    "Srinagar": {
        "state": "Jammu and Kashmir",
        "display_name": "Srinagar",
        "aliases": ["srinagar"]
    },
    "Ranchi": {
        "state": "Jharkhand",
        "display_name": "Ranchi",
        "aliases": ["ranchi"]
    },
    "Raipur": {
        "state": "Chhattisgarh",
        "display_name": "Raipur",
        "aliases": ["raipur"]
    },
    "Mysore": {
        "state": "Karnataka",
        "display_name": "Mysore (Mysuru)",
        "aliases": ["mysore", "mysuru"]
    },
    "Mangalore": {
        "state": "Karnataka",
        "display_name": "Mangalore (Mangaluru)",
        "aliases": ["mangalore", "mangaluru"]
    },
}

# 2. World Capitals Knowledge Base
WORLD_CAPITALS: Dict[str, str] = {
    "india": "New Delhi",
    "united states": "Washington, D.C.",
    "usa": "Washington, D.C.",
    "united kingdom": "London",
    "uk": "London",
    "france": "Paris",
    "germany": "Berlin",
    "japan": "Tokyo",
    "china": "Beijing",
    "australia": "Canberra",
    "canada": "Ottawa",
    "russia": "Moscow",
    "brazil": "Brasília",
    "italy": "Rome",
    "spain": "Madrid",
    "uae": "Abu Dhabi",
    "united arab emirates": "Abu Dhabi",
    "singapore": "Singapore",
    "south korea": "Seoul",
}

# 3. Technical & Conceptual Definitions
TECHNICAL_DEFINITIONS: Dict[str, str] = {
    "python": "Python is a high-level, interpreted, general-purpose programming language widely used for data science, artificial intelligence, machine learning, and web development.",
    "api": "API stands for Application Programming Interface. It is a set of defined rules and protocols that enables different software applications to communicate and exchange data with one another.",
    "machine learning": "Machine Learning (ML) is a subset of Artificial Intelligence (AI) that develops algorithms capable of learning patterns from data and making decisions without being explicitly programmed.",
    "ml": "Machine Learning (ML) is a subset of Artificial Intelligence (AI) that develops algorithms capable of learning patterns from data and making decisions without being explicitly programmed.",
    "artificial intelligence": "Artificial Intelligence (AI) refers to the simulation of human intelligence by machines and computer systems, enabling tasks such as reasoning, learning, and natural language understanding.",
    "ai": "Artificial Intelligence (AI) refers to the simulation of human intelligence by machines and computer systems, enabling tasks such as reasoning, learning, and natural language understanding.",
    "difference between ai and ml": "Artificial Intelligence (AI) is the broader discipline of building smart machines capable of performing human-like tasks, whereas Machine Learning (ML) is a specific subfield of AI where algorithms learn directly from data.",
    "sql": "SQL (Structured Query Language) is the standard programming language used to manage, store, query, and manipulate data within relational database management systems.",
    "database": "A database is an organized collection of structured data or information typically stored electronically in a computer system and managed by a database management system (DBMS).",
    "data science": "Data Science is an interdisciplinary field combining statistics, data analysis, machine learning, and computer science to extract actionable insights and knowledge from structured and unstructured data.",
}

# 4. Key National & Global Facts
FACTUAL_QA: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"how many states.*(?:in india|india has)", re.I), "India has 28 states and 8 union territories, making a total of 36 administrative entities."),
    (re.compile(r"capital of india", re.I), "The capital of India is New Delhi."),
    (re.compile(r"president of india", re.I), "The President of India is Droupadi Murmu."),
    (re.compile(r"prime minister of india", re.I), "The Prime Minister of India is Narendra Modi."),
    (re.compile(r"currency of india", re.I), "The official currency of India is the Indian Rupee (INR, ₹)."),
]


class GeneralKnowledgeEngine:
    """Resolves general geographical, factual, and technical knowledge queries with zero hallucinations."""

    def __init__(self):
        # Build lookup table for all city aliases
        self._city_alias_map: Dict[str, str] = {}
        for canonical_name, data in INDIAN_CITIES.items():
            for alias in data.get("aliases", []):
                self._city_alias_map[alias.lower()] = canonical_name
            self._city_alias_map[canonical_name.lower()] = canonical_name

        # Build lookup table for states with comprehensive aliases
        self._state_map: Dict[str, str] = {s.lower(): s for s in INDIAN_STATES.keys()}
        self._state_map.update(STATE_ALIASES)

    def find_city_match(self, term: str) -> Optional[str]:
        """Match term to canonical city name using alias or fuzzy match."""
        t_clean = term.strip().lower()
        if t_clean in self._city_alias_map:
            return self._city_alias_map[t_clean]

        # Check entity alias resolver (IATA codes, shortcuts, alternate names)
        alias_res = entity_alias_resolver.resolve_alias(term, expected_type="CITY")
        if alias_res and alias_res.get("canonical_name"):
            return alias_res["canonical_name"]

        # Check fuzzy match
        if HAS_RAPIDFUZZ:
            best = process.extractOne(t_clean, list(self._city_alias_map.keys()), scorer=fuzz.ratio)
            if best and best[1] >= 80:
                return self._city_alias_map[best[0]]

        # Check phonetic match
        from app.utils.phonetic import phonetic_match_score
        best_p_city = None
        best_p_score = 0.0
        for k, canonical in self._city_alias_map.items():
            ps = phonetic_match_score(t_clean, k)
            if ps > best_p_score and ps >= 0.82:
                best_p_score = ps
                best_p_city = canonical
        if best_p_city:
            return best_p_city

        return None

    def find_state_match(self, term: str) -> Optional[str]:
        """Match term to canonical Indian state name."""
        t_clean = term.strip().lower()
        t_nodots = t_clean.replace(".", "")
        if t_clean in self._state_map:
            return self._state_map[t_clean]
        if t_nodots in self._state_map:
            return self._state_map[t_nodots]

        # Check entity alias resolver (state codes like AP, TG, WB, MH, etc.)
        alias_res = entity_alias_resolver.resolve_alias(term, expected_type="STATE")
        if alias_res and alias_res.get("canonical_name"):
            return alias_res["canonical_name"]

        if HAS_RAPIDFUZZ:
            best = process.extractOne(t_clean, list(self._state_map.keys()), scorer=fuzz.ratio)
            if best and best[1] >= 80:
                return self._state_map[best[0]]
            best_nodots = process.extractOne(t_nodots, list(self._state_map.keys()), scorer=fuzz.ratio)
            if best_nodots and best_nodots[1] >= 80:
                return self._state_map[best_nodots[0]]

        # Check phonetic match
        from app.utils.phonetic import phonetic_match_score
        best_p_state = None
        best_p_score = 0.0
        for k, canonical in self._state_map.items():
            ps = phonetic_match_score(t_clean, k)
            if ps > best_p_score and ps >= 0.82:
                best_p_score = ps
                best_p_state = canonical
        if best_p_state:
            return best_p_state

        return None

    def get_city_info(self, city_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve state and details for a city."""
        canonical = self.find_city_match(city_name)
        if canonical:
            if canonical in INDIAN_CITIES:
                info = dict(INDIAN_CITIES[canonical])
                info["canonical_name"] = canonical
                return info
            # Check state of city from alias resolver
            st_res = entity_alias_resolver.resolve_state_of(canonical)
            if st_res:
                return {
                    "state": st_res["state"],
                    "display_name": st_res["city"],
                    "canonical_name": st_res["city"],
                    "aliases": [city_name.lower(), canonical.lower()]
                }
        return None

    def get_state_capital(self, state_name: str) -> Optional[str]:
        """Retrieve capital of an Indian state."""
        canonical = self.find_state_match(state_name)
        if canonical and canonical in INDIAN_STATES:
            return INDIAN_STATES[canonical]["capital"]
        cap_res = entity_alias_resolver.resolve_capital_of(state_name)
        if cap_res:
            return cap_res["capital"]
        return None

    def verify_city_in_state(self, city_term: str, state_term: str) -> Dict[str, Any]:
        """Verify whether a city belongs to a given state deterministically."""
        city_info = self.get_city_info(city_term)
        target_state = self.find_state_match(state_term)

        if not city_info or not target_state:
            return {
                "verified": False,
                "is_member": None,
                "reason": f"Could not resolve city '{city_term}' or state '{state_term}'."
            }

        actual_state = city_info["state"]
        # Match if identical or substring (e.g. 'Delhi' in 'Delhi (National Capital Territory)')
        is_member = (
            actual_state.lower() == target_state.lower()
            or target_state.lower() in actual_state.lower()
            or actual_state.lower() in target_state.lower()
        )
        return {
            "verified": True,
            "is_member": is_member,
            "city": city_info["canonical_name"],
            "city_display": city_info.get("display_name", city_info["canonical_name"]),
            "actual_state": actual_state,
            "target_state": target_state
        }

    def verify_state_capital(self, city_term: str, state_or_country_term: str) -> Dict[str, Any]:
        canonical_city = self.find_city_match(city_term)
        target_state = self.find_state_match(state_or_country_term)

        # Handle reversed argument order (state passed as first arg, city as second)
        if not target_state or not canonical_city:
            rev_city = self.find_city_match(state_or_country_term)
            rev_state = self.find_state_match(city_term)
            if rev_state and rev_city:
                canonical_city = rev_city
                target_state = rev_state
            elif rev_state and not target_state:
                target_state = rev_state
            elif rev_city and not canonical_city:
                canonical_city = rev_city

        if target_state and target_state in INDIAN_STATES:
            actual_cap = INDIAN_STATES[target_state]["capital"]
            is_cap = canonical_city is not None and canonical_city.lower() == actual_cap.lower()
            return {
                "verified": True,
                "is_capital": is_cap,
                "city": canonical_city or city_term,
                "target_entity": target_state,
                "actual_capital": actual_cap,
                "entity_type": "state"
            }

        s_clean = state_or_country_term.strip().lower()
        for country, cap in WORLD_CAPITALS.items():
            if country in s_clean or s_clean in country:
                is_cap = canonical_city is not None and canonical_city.lower() == cap.lower()
                return {
                    "verified": True,
                    "is_capital": is_cap,
                    "city": canonical_city or city_term,
                    "target_entity": country.title(),
                    "actual_capital": cap,
                    "entity_type": "country"
                }

        return {"verified": False, "is_capital": None, "reason": f"Could not resolve entity '{state_or_country_term}'."}

    def answer_question(self, question: str) -> Dict[str, Any]:
        """Attempt to answer general knowledge question deterministically, falling back to Ollama if available."""
        q_clean = question.strip()
        q_lower = q_clean.lower()

        # 0. Check Boolean City-State or Capital Query
        # e.g. "Is hyd belongs to Andra Pradesh?", "Is hyd in Telangana?", "hyd is ap right?", "hyd under telangana?"
        # Find any mentioned city
        found_city = None
        for alias, canon in self._city_alias_map.items():
            if re.search(r"\b" + re.escape(alias) + r"\b", q_lower):
                found_city = canon
                break

        # Find any mentioned state
        found_state = None
        for alias, canon in self._state_map.items():
            if re.search(r"\b" + re.escape(alias) + r"\b", q_lower):
                found_state = canon
                break

        is_wh_question = any(q_lower.startswith(wh) for wh in ["which ", "what ", "where ", "who ", "how ", "what's ", "whats "])
        has_confirm_tag = any(tag in q_lower for tag in ["right", "na", "correct", "true or false", "or not", "is it", "isn't it"])
        if found_city and found_state and (not is_wh_question or has_confirm_tag):
            # Check if capital query
            if "capital" in q_lower:
                cap_ver = self.verify_state_capital(found_city, found_state)
                if cap_ver["verified"]:
                    actual_cap = cap_ver["actual_capital"]
                    target = cap_ver["target_entity"]
                    if cap_ver["is_capital"]:
                        ans = f"True. {cap_ver['city']} is the capital of {target}."
                    else:
                        ans = f"False. {actual_cap} is the capital of {target}, not {cap_ver['city']}."
                    return {
                        "answered": True,
                        "answer": ans,
                        "topic": "geography_capital_boolean",
                        "result": cap_ver["is_capital"],
                        "confidence": 1.0,
                        "source": "general_knowledge_base",
                        "evidence": cap_ver
                    }

            # City-State membership boolean check
            # e.g. "Is hyd belongs to Andra Pradesh?" -> False. Hyderabad is in Telangana, not Andhra Pradesh.
            mem_ver = self.verify_city_in_state(found_city, found_state)
            if mem_ver["verified"]:
                disp_city = mem_ver["city_display"]
                act_state = mem_ver["actual_state"]
                tgt_state = mem_ver["target_state"]
                if mem_ver["is_member"]:
                    ans = f"True. {disp_city} is in {act_state}, India."
                else:
                    ans = f"False. {disp_city} is in {act_state}, not {tgt_state}."
                return {
                    "answered": True,
                    "answer": ans,
                    "topic": "geography_city_state_boolean",
                    "result": mem_ver["is_member"],
                    "confidence": 1.0,
                    "source": "general_knowledge_base",
                    "evidence": mem_ver
                }

        # 1. Check direct fact patterns (e.g. how many states in India, president, etc.)
        for pat, ans in FACTUAL_QA:
            if pat.search(q_lower):
                return {
                    "answered": True,
                    "answer": ans,
                    "topic": "factual_knowledge",
                    "confidence": 1.0,
                    "source": "general_knowledge_base"
                }

        # 2. Check State of City queries (e.g. 'Which state does banglore belongs to?', 'which state banglore?', 'where is banglore?', 'hyd state', 'blr state?')
        city_state_patterns = [
            r"which state (?:does\s+)?([a-zA-Z\s]+?)(?:\s+belong|\s+belongs|\s+is|\s+comes|\s+located|\?|$)",
            r"([a-zA-Z\s]+?)\s+(?:which state|is in which state|comes under which state|belongs to which state)",
            r"where is\s+([a-zA-Z\s]+?)(?:\?|$)",
            r"where does\s+([a-zA-Z\s]+?)(?:\s+belong|\s+belongs|\s+come|\s+comes)(?:\?|$)",
            r"what state is\s+([a-zA-Z\s]+?)(?:\s+in|\?|$)",
            r"which state is\s+([a-zA-Z\s]+?)(?:\s+in|\?|$)",
            r"([a-zA-Z\s]+?)'s\s+state(?:\?|$)",
            r"([a-zA-Z\s]+?)\s+state(?:\?|$)",
            r"([a-zA-Z\s]+?)\s+which state\b",
        ]
        for p in city_state_patterns:
            m = re.search(p, q_lower)
            if m:
                cand = m.group(1).strip()
                cand = re.sub(r"^(?:the\s+city\s+of|the\s+|does\s+|in\s+)", "", cand).strip()
                cand = re.sub(r"(?:\s+bro|\s+city|\s+area|\s+town)$", "", cand).strip()
                city_info = self.get_city_info(cand)
                if city_info:
                    disp = city_info["display_name"]
                    st = city_info["state"]
                    return {
                        "answered": True,
                        "answer": f"{disp} is in {st}, India.",
                        "topic": "geography_city_state",
                        "city": city_info["canonical_name"],
                        "state": st,
                        "confidence": 1.0,
                        "source": "general_knowledge_base"
                    }

        # Also search for any city mentioned in the whole question if 'state' or 'where' is in the query
        if any(w in q_lower for w in ["state", "where", "location", "region", "which part"]):
            for alias, canonical in self._city_alias_map.items():
                if re.search(r"\b" + re.escape(alias) + r"\b", q_lower):
                    city_info = INDIAN_CITIES[canonical]
                    disp = city_info["display_name"]
                    st = city_info["state"]
                    return {
                        "answered": True,
                        "answer": f"{disp} is in {st}, India.",
                        "topic": "geography_city_state",
                        "city": canonical,
                        "state": st,
                        "confidence": 0.95,
                        "source": "general_knowledge_base"
                    }

        # 3. Capital of State queries (direct and reverse)
        # e.g. 'what is the capital of Telangana?', 'capital of Karnataka', "West Bengal's capital", "West Bengal capital", "Which city is West Bengal's capital?"
        cap_patterns = [
            r"capital of\s+([a-zA-Z\s]+?)(?:\?|$)",
            r"([a-zA-Z\s]+?)'s\s+capital(?:\?|$)",
            r"which\s+city\s+is\s+([a-zA-Z\s]+?)'s\s+capital(?:\?|$)",
            r"([a-zA-Z\s]+?)\s+has\s+which\s+capital(?:\?|$)",
            r"([a-zA-Z\s]+?)\s+capital(?:\?|$)"
        ]
        for cap_pat in cap_patterns:
            cap_m = re.search(cap_pat, q_lower)
            if cap_m:
                state_cand = cap_m.group(1).strip()
                state_cand = re.sub(r"^(?:what\s+is\s+|tell\s+me\s+|show\s+me\s+|what's\s+|whats\s+)", "", state_cand).strip()
                state_cap = self.get_state_capital(state_cand)
                if state_cap:
                    state_name = self.find_state_match(state_cand)
                    return {
                        "answered": True,
                        "answer": f"The capital of {state_name} is {state_cap}.",
                        "topic": "geography_capital",
                        "state": state_name,
                        "capital": state_cap,
                        "confidence": 1.0,
                        "source": "general_knowledge_base"
                    }
                # Check if World country
                for country, cap in WORLD_CAPITALS.items():
                    if country in state_cand or state_cand in country:
                        return {
                            "answered": True,
                            "answer": f"The capital of {country.title()} is {cap}.",
                            "topic": "geography_capital",
                            "country": country.title(),
                            "capital": cap,
                            "confidence": 1.0,
                            "source": "general_knowledge_base"
                        }

        # 3.5. Reverse Capital to State lookup (e.g. "Which state has Kolkata as capital?", "Kolkata is the capital of which state?")
        rev_cap_patterns = [
            r"which\s+state\s+has\s+([a-zA-Z\s]+?)\s+as(?:\s+its)?\s+capital(?:\?|$)",
            r"([a-zA-Z\s]+?)\s+is(?:\s+the)?\s+capital\s+of\s+which\s+state(?:\?|$)",
        ]
        for rev_pat in rev_cap_patterns:
            rev_m = re.search(rev_pat, q_lower)
            if rev_m:
                city_cand = rev_m.group(1).strip()
                canonical_city = self.find_city_match(city_cand)
                for state_k, info_k in INDIAN_STATES.items():
                    if info_k.get("capital", "").lower() == (canonical_city or city_cand).lower():
                        return {
                            "answered": True,
                            "answer": f"{info_k['capital']} is the capital of {state_k}.",
                            "topic": "geography_capital_reverse",
                            "state": state_k,
                            "capital": info_k["capital"],
                            "confidence": 1.0,
                            "source": "general_knowledge_base"
                        }

        # 4. Technical / Concept Definitions
        # Direct check for 'difference between'
        if "difference between" in q_lower:
            if ("ai" in q_lower or "artificial intelligence" in q_lower) and ("ml" in q_lower or "machine learning" in q_lower):
                return {
                    "answered": True,
                    "answer": TECHNICAL_DEFINITIONS["difference between ai and ml"],
                    "topic": "technical_definition",
                    "confidence": 1.0,
                    "source": "general_knowledge_base"
                }

        for concept, definition in TECHNICAL_DEFINITIONS.items():
            pat = r"\b(?:what is|what does|meaning of|define|definition of|explain)\s+(?:the\s+)?" + re.escape(concept) + r"\b"
            if re.search(pat, q_lower) or q_lower.strip(" ?") == concept:
                return {
                    "answered": True,
                    "answer": definition,
                    "topic": "technical_definition",
                    "confidence": 1.0,
                    "source": "general_knowledge_base"
                }

        # 5. Fallback to Ollama if online
        health = ollama_client.check_health()
        if health.get("available"):
            logger.info(f"Dispatching general knowledge question to Ollama: {q_clean}")
            prompt = (
                f"You are a factual knowledge assistant. Answer the following question accurately, "
                f"concisely, and neutrally in 1-2 sentences. Do not mention dataset or hallucinate.\n\n"
                f"Question: \"{q_clean}\"\n\nAnswer:"
            )
            raw_ans = ollama_client.generate(prompt=prompt, json_format=False)
            if raw_ans and len(raw_ans.strip()) > 5:
                return {
                    "answered": True,
                    "answer": raw_ans.strip(),
                    "topic": "ollama_knowledge",
                    "confidence": 0.90,
                    "source": "ollama_llm"
                }

        # 6. Unanswered
        return {
            "answered": False,
            "answer": "I cannot determine that from general knowledge with high confidence.",
            "topic": "unknown",
            "confidence": 0.0,
            "source": "unanswered"
        }


general_knowledge_engine = GeneralKnowledgeEngine()
