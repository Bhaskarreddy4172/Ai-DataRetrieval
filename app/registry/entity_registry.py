"""Entity Registry: Canonical entity store and alias resolution service."""

from typing import Any, Dict, List, Optional
from app.database.repositories import entity_repo
from app.utils.fuzzy_match import fuzzy_match


class EntityRegistry:
    """Manages canonical entities (States, Capitals, Villages) and multi-lingual/code aliases."""

    def __init__(self):
        self.repo = entity_repo

    def resolve_entity(self, term: str) -> Optional[str]:
        """Resolves alias, abbreviation, or exact name to canonical entity name."""
        return self.repo.resolve_alias(term)

    def search_entities(
        self,
        query: str,
        entity_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Search entities with optional type filter."""
        results = self.repo.search_entities(query=query, limit=limit * 2)
        if entity_type:
            results = [r for r in results if r["entity_type"].lower() == entity_type.lower()]
        return results[:limit]

    def fuzzy_match_entity(self, term: str, threshold: float = 0.8) -> Optional[str]:
        """Fuzzy match term against all canonical entities."""
        all_ents = self.repo.search_entities(limit=2000)
        names = [e["name"] for e in all_ents]
        match, score = fuzzy_match(term, names, threshold=threshold)
        return match if match else None


entity_registry = EntityRegistry()
