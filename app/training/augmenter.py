"""Linguistic question augmenter creating typos, abbreviations, slang, broken grammar, and paraphrases."""

import random
import re
from typing import List

SLANG_SUBSTITUTIONS = {
    "highest salary": ["biggest paycheck", "making bank", "top earner", "fat stack", "most cash"],
    "lowest salary": ["least paid", "smallest paycheck", "who makes peanuts"],
    "employee": ["dude", "guy", "worker", "person", "staffer"],
    "department": ["dept", "team", "division", "unit"],
    "hyderabad": ["hyd", "hyderbad", "hydrabad"],
    "bengaluru": ["blr", "bangalore", "banglore"],
    "mumbai": ["mum", "bombay"],
    "chennai": ["madras", "chn"],
    "delhi": ["del", "dilli"],
    "performance": ["perf", "stars", "rating"],
    "manager": ["boss", "mgr", "head"]
}

FILLER_PREFIXES = [
    "hey can you tell me",
    "please show",
    "i want to know",
    "yo",
    "could you find",
    "give me",
    "display",
    "who is",
    "list"
]


class LinguisticAugmenter:
    """Applies linguistic perturbations to simulate natural unseen customer questions."""

    def __init__(self, seed: int = 42):
        random.seed(seed)

    def introduce_typo(self, text: str) -> str:
        """Introduce realistic typographical errors."""
        words = text.split()
        if not words:
            return text
        idx = random.randint(0, len(words) - 1)
        w = words[idx]
        if len(w) > 4:
            vowels = [i for i, c in enumerate(w) if c in "aeiou"]
            if vowels:
                drop_idx = random.choice(vowels)
                w = w[:drop_idx] + w[drop_idx+1:]
            else:
                w = w[:-1]
            words[idx] = w
        return " ".join(words)

    def add_slang_and_abbrev(self, text: str) -> str:
        """Substitute formal phrases with colloquialisms or abbreviations."""
        augmented = text
        for formal, slangs in SLANG_SUBSTITUTIONS.items():
            if formal in augmented.lower():
                choice = random.choice(slangs)
                augmented = re.sub(rf"\b{re.escape(formal)}\b", choice, augmented, flags=re.IGNORECASE)
        return augmented

    def add_filler(self, text: str) -> str:
        """Add conversational prefix or informal wrapper."""
        prefix = random.choice(FILLER_PREFIXES)
        return f"{prefix} {text.lower()}".strip()

    def augment(self, text: str) -> List[str]:
        """Generate multiple linguistic variations of a seed query."""
        variations = [text]
        variations.append(self.add_slang_and_abbrev(text))
        variations.append(self.add_filler(text))
        variations.append(self.introduce_typo(text))
        variations.append(self.add_filler(self.add_slang_and_abbrev(text)))
        return list(dict.fromkeys(variations))


linguistic_augmenter = LinguisticAugmenter()
