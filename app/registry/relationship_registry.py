"""Relationship Registry: Manages parent-child and relational links between datasets."""

from typing import Any, Dict, List, Optional
from app.database.connection import db_manager
from app.database.models import DatasetRelationshipModel


class RelationshipRegistry:
    """Provides lookup of parent-child dataset connections."""

    @staticmethod
    def get_child_datasets(parent_dataset_id: str) -> List[Dict[str, Any]]:
        session = db_manager.get_session()
        try:
            records = session.query(DatasetRelationshipModel).filter_by(parent_dataset_id=parent_dataset_id).all()
            return [
                {
                    "relationship_id": r.relationship_id,
                    "parent_dataset_id": r.parent_dataset_id,
                    "child_dataset_id": r.child_dataset_id,
                    "parent_column": r.parent_column,
                    "child_column": r.child_column,
                    "relationship_type": r.relationship_type,
                }
                for r in records
            ]
        finally:
            session.close()

    @staticmethod
    def get_parent_relationship(child_dataset_id: str) -> Optional[Dict[str, Any]]:
        session = db_manager.get_session()
        try:
            r = session.query(DatasetRelationshipModel).filter_by(child_dataset_id=child_dataset_id).first()
            if not r:
                return None
            return {
                "relationship_id": r.relationship_id,
                "parent_dataset_id": r.parent_dataset_id,
                "child_dataset_id": r.child_dataset_id,
                "parent_column": r.parent_column,
                "child_column": r.child_column,
                "relationship_type": r.relationship_type,
            }
        finally:
            session.close()


relationship_registry = RelationshipRegistry()
