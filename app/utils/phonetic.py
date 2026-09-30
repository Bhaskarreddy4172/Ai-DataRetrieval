"""Pure Python Soundex, Double Metaphone, and Phonetic Candidate Matcher.

Designed for robust resolution of phonetic variants, sound-alikes, and typo-distorted
names and entities (e.g. Cikkim -> Sikkim, Kalkata -> Kolkata, Haiderabad -> Hyderabad).
"""

import re
from typing import Dict, List, Optional, Set, Tuple


def soundex(word: str) -> str:
    """Calculate American Soundex code for a word.
    
    Returns standard 4-character code (e.g., 'H361' for 'Hyderabad').
    """
    if not word:
        return ""

    w = re.sub(r"[^A-Za-z]", "", word.upper())
    if not w:
        return ""

    first_letter = w[0]

    # Mapping table:
    # 1: B, F, P, V
    # 2: C, G, J, K, Q, S, X, Z
    # 3: D, T
    # 4: L
    # 5: M, N
    # 6: R
    # 0 / ignored: A, E, I, O, U, Y, H, W
    char_map = {
        "B": "1", "F": "1", "P": "1", "V": "1",
        "C": "2", "G": "2", "J": "2", "K": "2", "Q": "2", "S": "2", "X": "2", "Z": "2",
        "D": "3", "T": "3",
        "L": "4",
        "M": "5", "N": "5",
        "R": "6",
    }

    digits = [first_letter]
    prev_code = char_map.get(first_letter, "0")

    for char in w[1:]:
        code = char_map.get(char, "0")
        if code != "0":
            if code != prev_code:
                digits.append(code)
            prev_code = code
        else:
            # Vowels reset repetition blocker in standard soundex, except H and W
            if char not in ("H", "W"):
                prev_code = "0"

    # Pad or truncate to 4 characters
    soundex_code = "".join(digits[:4]).ljust(4, "0")
    return soundex_code


def double_metaphone(word: str) -> Tuple[str, str]:
    """Calculate Lawrence Philips' Double Metaphone (Primary, Alternate) codes.
    
    Provides rich phonetic representation handling silent letters, Germanic,
    Slavic, Romance, and Anglo-Indian phonetic variations.
    """
    if not word:
        return ("", "")

    w = re.sub(r"[^A-Za-z]", "", word.upper())
    if not w:
        return ("", "")

    # Normalize initial silent/special combinations
    if w.startswith(("GN", "KN", "PN", "WR", "PS")):
        w = w[1:]
    elif w.startswith("X"):
        w = "S" + w[1:]

    primary: List[str] = []
    secondary: List[str] = []

    length = len(w)
    idx = 0

    def char_at(pos: int) -> str:
        if 0 <= pos < length:
            return w[pos]
        return ""

    def substr(pos: int, count: int) -> str:
        return w[pos : pos + count]

    while idx < length:
        ch = char_at(idx)

        # Vowels at word start get 'A'
        if ch in ("A", "E", "I", "O", "U", "Y"):
            if idx == 0:
                primary.append("A")
                secondary.append("A")
            idx += 1
            continue

        # B
        if ch == "B":
            primary.append("P")
            secondary.append("P")
            idx += 2 if char_at(idx + 1) == "B" else 1
            continue

        # C
        if ch == "C":
            # CI, CE, CY -> S (e.g. Cikkim -> S, City -> S)
            if substr(idx, 2) in ("CI", "CE", "CY"):
                primary.append("S")
                secondary.append("S")
                idx += 2
                continue
            # CH
            if substr(idx, 2) == "CH":
                primary.append("X")
                secondary.append("K")
                idx += 2
                continue
            # CK / CG / CQ
            if substr(idx, 2) in ("CK", "CG", "CQ"):
                primary.append("K")
                secondary.append("K")
                idx += 2
                continue
            # CC
            if substr(idx, 2) == "CC":
                if char_at(idx + 2) in ("I", "E", "Y"):
                    primary.append("KS")
                    secondary.append("KS")
                    idx += 3
                else:
                    primary.append("K")
                    secondary.append("K")
                    idx += 2
                continue
            primary.append("K")
            secondary.append("K")
            idx += 2 if char_at(idx + 1) == "C" else 1
            continue

        # D
        if ch == "D":
            if substr(idx, 2) == "DG":
                if char_at(idx + 2) in ("I", "E", "Y"):
                    primary.append("J")
                    secondary.append("J")
                    idx += 3
                else:
                    primary.append("TK")
                    secondary.append("TK")
                    idx += 2
                continue
            if substr(idx, 2) in ("DT", "DD"):
                primary.append("T")
                secondary.append("T")
                idx += 2
                continue
            primary.append("T")
            secondary.append("T")
            idx += 1
            continue

        # F
        if ch == "F":
            primary.append("F")
            secondary.append("F")
            idx += 2 if char_at(idx + 1) == "F" else 1
            continue

        # G
        if ch == "G":
            if char_at(idx + 1) == "H":
                if idx > 0 and char_at(idx - 1) not in ("A", "E", "I", "O", "U", "Y"):
                    primary.append("K")
                    secondary.append("K")
                idx += 2
                continue
            if substr(idx, 2) == "GN":
                primary.append("N")
                secondary.append("N")
                idx += 2
                continue
            if char_at(idx + 1) in ("I", "E", "Y"):
                primary.append("J")
                secondary.append("K")
                idx += 2
                continue
            primary.append("K")
            secondary.append("K")
            idx += 2 if char_at(idx + 1) == "G" else 1
            continue

        # H
        if ch == "H":
            # Keep H only before vowel and not preceded by vowel
            if char_at(idx + 1) in ("A", "E", "I", "O", "U", "Y") and (idx == 0 or char_at(idx - 1) not in ("A", "E", "I", "O", "U", "Y")):
                primary.append("H")
                secondary.append("H")
            idx += 1
            continue

        # J
        if ch == "J":
            primary.append("J")
            secondary.append("H")
            idx += 2 if char_at(idx + 1) == "J" else 1
            continue

        # K
        if ch == "K":
            primary.append("K")
            secondary.append("K")
            idx += 2 if char_at(idx + 1) == "K" else 1
            continue

        # L
        if ch == "L":
            primary.append("L")
            secondary.append("L")
            idx += 2 if char_at(idx + 1) == "L" else 1
            continue

        # M
        if ch == "M":
            primary.append("M")
            secondary.append("M")
            idx += 2 if char_at(idx + 1) == "M" else 1
            continue

        # N
        if ch == "N":
            primary.append("N")
            secondary.append("N")
            idx += 2 if char_at(idx + 1) == "N" else 1
            continue

        # P
        if ch == "P":
            if char_at(idx + 1) == "H":
                primary.append("F")
                secondary.append("F")
                idx += 2
                continue
            primary.append("P")
            secondary.append("P")
            idx += 2 if char_at(idx + 1) == "P" else 1
            continue

        # Q
        if ch == "Q":
            primary.append("K")
            secondary.append("K")
            idx += 2 if char_at(idx + 1) == "Q" else 1
            continue

        # R
        if ch == "R":
            primary.append("R")
            secondary.append("R")
            idx += 2 if char_at(idx + 1) == "R" else 1
            continue

        # S
        if ch == "S":
            if substr(idx, 2) == "SH":
                primary.append("X")
                secondary.append("X")
                idx += 2
                continue
            if substr(idx, 3) in ("SIO", "SIA"):
                primary.append("X")
                secondary.append("S")
                idx += 3
                continue
            primary.append("S")
            secondary.append("S")
            idx += 2 if char_at(idx + 1) == "S" else 1
            continue

        # T
        if ch == "T":
            if substr(idx, 2) == "TH":
                primary.append("0")
                secondary.append("T")
                idx += 2
                continue
            if substr(idx, 3) in ("TIA", "TCH"):
                primary.append("X")
                secondary.append("X")
                idx += 3
                continue
            primary.append("T")
            secondary.append("T")
            idx += 2 if char_at(idx + 1) == "T" else 1
            continue

        # V
        if ch == "V":
            primary.append("F")
            secondary.append("F")
            idx += 2 if char_at(idx + 1) == "V" else 1
            continue

        # W
        if ch == "W":
            if char_at(idx + 1) in ("A", "E", "I", "O", "U"):
                primary.append("A")
                secondary.append("F")
            idx += 1
            continue

        # X
        if ch == "X":
            primary.append("KS")
            secondary.append("KS")
            idx += 2 if char_at(idx + 1) == "X" else 1
            continue

        # Z
        if ch == "Z":
            primary.append("S")
            secondary.append("S")
            idx += 2 if char_at(idx + 1) == "Z" else 1
            continue

        idx += 1

    prim_str = "".join(primary)[:6]
    sec_str = "".join(secondary)[:6]
    return (prim_str, sec_str or prim_str)


def normalize_indian_phonetics(text: str) -> str:
    """Normalize common phonetic and transliteration variations in Indian names and places.
    
    Examples:
    - Cikkim -> Sikkim
    - Kalkata -> Kolkata
    - Bengluru -> Bengaluru
    - Haiderabad / Haydarabad -> Hyderabad
    - Aandhra -> Andhra
    - Sikim -> Sikkim
    """
    t = text.lower()

    # Vowel transliteration smoothing
    t = re.sub(r"aa+", "a", t)
    t = re.sub(r"ee+", "i", t)
    t = re.sub(r"oo+", "u", t)
    t = re.sub(r"ou+", "au", t)
    t = re.sub(r"ay|ey|ei", "ai", t)
    t = re.sub(r"der\b|der", "dar", t)

    # Initial sound-alikes: 'c' before 'i'/'e' sounds like 's' (e.g. Cikkim -> Sikkim)
    if t.startswith("ci") or t.startswith("ce"):
        t = "s" + t[1:]

    # Aspirated consonant softening
    t = re.sub(r"bh", "b", t)
    t = re.sub(r"dh", "d", t)
    t = re.sub(r"gh", "g", t)
    t = re.sub(r"kh", "k", t)
    t = re.sub(r"ph", "f", t)
    t = re.sub(r"th", "t", t)

    # Intervocalic / glide variants
    t = re.sub(r"v", "w", t)

    # Double consonant reduction
    t = re.sub(r"([a-z])\1+", r"\1", t)

    return t


class PhoneticIndex:
    """Inverted index mapping Soundex and Double Metaphone codes to candidate tokens."""

    def __init__(self):
        self.soundex_map: Dict[str, Set[str]] = {}
        self.metaphone_primary_map: Dict[str, Set[str]] = {}
        self.metaphone_secondary_map: Dict[str, Set[str]] = {}
        self.normalized_map: Dict[str, Set[str]] = {}
        self.all_values: Set[str] = set()

    def add_term(self, term: str) -> None:
        """Index a term by all phonetic keys."""
        if not term:
            return
        clean = term.strip()
        if not clean:
            return

        self.all_values.add(clean)

        # Index tokens for multi-word phrases as well as the full phrase
        tokens = [clean] + [t for t in re.split(r"[\s\-_/]+", clean) if len(t) >= 3]

        for tok in tokens:
            # Soundex
            sx = soundex(tok)
            if sx:
                self.soundex_map.setdefault(sx, set()).add(clean)

            # Double Metaphone
            p, s = double_metaphone(tok)
            if p:
                self.metaphone_primary_map.setdefault(p, set()).add(clean)
            if s:
                self.metaphone_secondary_map.setdefault(s, set()).add(clean)

            # Indian phonetic normalization key
            norm_key = normalize_indian_phonetics(tok)
            if norm_key:
                self.normalized_map.setdefault(norm_key, set()).add(clean)

    def get_candidates(self, query_term: str) -> Set[str]:
        """Retrieve all candidate terms matching query phonetically."""
        if not query_term:
            return set()

        q_clean = query_term.strip()
        candidates: Set[str] = set()

        # Query Soundex
        sx = soundex(q_clean)
        if sx in self.soundex_map:
            candidates.update(self.soundex_map[sx])

        # Query Double Metaphone
        p, s = double_metaphone(q_clean)
        if p in self.metaphone_primary_map:
            candidates.update(self.metaphone_primary_map[p])
        if s in self.metaphone_secondary_map:
            candidates.update(self.metaphone_secondary_map[s])

        # Indian phonetic normalization
        norm_key = normalize_indian_phonetics(q_clean)
        if norm_key in self.normalized_map:
            candidates.update(self.normalized_map[norm_key])

        return candidates


def phonetic_match_score(query: str, target: str) -> float:
    """Calculate composite phonetic similarity score between 0.0 and 1.0.
    
    Blends Soundex match, Double Metaphone match, Indian phonetic match,
    and Levenshtein edit distance.
    """
    import difflib

    q = query.strip().lower()
    tgt = target.strip().lower()

    if q == tgt:
        return 1.0

    score = 0.0

    # 1. Indian phonetic normalized equality
    if normalize_indian_phonetics(q) == normalize_indian_phonetics(tgt):
        score = max(score, 0.92)

    # 2. Double Metaphone primary/secondary match
    qp, qs = double_metaphone(q)
    tp, ts = double_metaphone(tgt)

    if qp and tp and qp == tp:
        score = max(score, 0.88)
    elif (qp and ts and qp == ts) or (qs and tp and qs == tp):
        score = max(score, 0.82)

    # 3. Soundex match
    if soundex(q) == soundex(tgt):
        score = max(score, 0.80)

    # 4. Levenshtein ratio blend
    lev_ratio = difflib.SequenceMatcher(None, q, tgt).ratio()
    if score > 0:
        # Boost phonetic score with string proximity
        final_score = (score * 0.65) + (lev_ratio * 0.35)
    else:
        final_score = lev_ratio

    return round(final_score, 3)
