"""Database Repositories: Clean data access layer for all models."""

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from sqlalchemy import desc, func, or_, text
from sqlalchemy.orm import Session

from app.database.connection import db_manager
from app.database.models import (
    DatasetColumnModel, DatasetModel, DatasetRelationshipModel,
    EntityAliasModel, EntityModel, RAGDocumentModel, StateDataModel,
    VillageDataModel, ConversationSessionModel, ConversationMessageModel,
    QueryLogModel, ValidationResultModel
)
from app.utils.logger import logger


class DatasetRepository:
    """Repository managing dataset catalog, columns, and relationships."""

    def __init__(self, session: Optional[Session] = None):
        self._session = session

    def _get_session(self) -> Session:
        return self._session or db_manager.get_session()

    def upsert_dataset(self, data: Dict[str, Any]) -> DatasetModel:
        session = self._get_session()
        try:
            ds = session.query(DatasetModel).filter_by(dataset_id=data["dataset_id"]).first()
            if not ds:
                ds = DatasetModel(**data)
                session.add(ds)
            else:
                for k, v in data.items():
                    if k != "dataset_id":
                        setattr(ds, k, v)
                ds.updated_at = datetime.now(timezone.utc)
            session.commit()
            return ds
        except Exception:
            session.rollback()
            raise
        finally:
            if not self._session:
                session.close()

    def get_dataset(self, dataset_id: str) -> Optional[DatasetModel]:
        session = self._get_session()
        try:
            return session.query(DatasetModel).filter_by(dataset_id=dataset_id).first()
        finally:
            if not self._session:
                session.close()

    def list_datasets(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        session = self._get_session()
        try:
            q = session.query(DatasetModel)
            if status:
                q = q.filter_by(status=status)
            records = q.all()
            return [
                {
                    "dataset_id": r.dataset_id,
                    "dataset_name": r.dataset_name,
                    "file_name": r.file_name,
                    "file_path": r.file_path,
                    "file_type": r.file_type,
                    "dataset_type": r.dataset_type,
                    "parent_entity": r.parent_entity,
                    "child_entity": r.child_entity,
                    "row_count": r.row_count,
                    "column_count": r.column_count,
                    "content_hash": r.content_hash,
                    "version": r.version,
                    "status": r.status,
                    "updated_at": str(r.updated_at),
                }
                for r in records
            ]
        finally:
            if not self._session:
                session.close()

    def save_columns(self, dataset_id: str, columns: List[Dict[str, Any]]) -> None:
        session = self._get_session()
        try:
            session.query(DatasetColumnModel).filter_by(dataset_id=dataset_id).delete()
            for col in columns:
                cm = DatasetColumnModel(
                    column_id=f"{dataset_id}_{col.get('normalized_name', col.get('original_name'))}",
                    dataset_id=dataset_id,
                    original_name=col["original_name"],
                    normalized_name=col["normalized_name"],
                    semantic_type=col.get("semantic_type", "TEXT"),
                    data_type=col.get("data_type", "string"),
                    is_numeric=col.get("is_numeric", False),
                    is_filterable=col.get("is_filterable", True),
                    is_aggregatable=col.get("is_aggregatable", False),
                )
                session.add(cm)
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            if not self._session:
                session.close()

    def get_columns(self, dataset_id: str) -> List[Dict[str, Any]]:
        session = self._get_session()
        try:
            cols = session.query(DatasetColumnModel).filter_by(dataset_id=dataset_id).all()
            return [
                {
                    "column_id": c.column_id,
                    "original_name": c.original_name,
                    "normalized_name": c.normalized_name,
                    "semantic_type": c.semantic_type,
                    "data_type": c.data_type,
                    "is_numeric": c.is_numeric,
                    "is_filterable": c.is_filterable,
                    "is_aggregatable": c.is_aggregatable,
                }
                for c in cols
            ]
        finally:
            if not self._session:
                session.close()

    def save_relationship(self, rel: Dict[str, Any]) -> None:
        session = self._get_session()
        try:
            existing = session.query(DatasetRelationshipModel).filter_by(relationship_id=rel["relationship_id"]).first()
            if not existing:
                session.add(DatasetRelationshipModel(**rel))
            else:
                for k, v in rel.items():
                    setattr(existing, k, v)
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            if not self._session:
                session.close()


class StateVillageRepository:
    """Repository managing tabular state and village data."""

    def __init__(self, session: Optional[Session] = None):
        self._session = session

    def _get_session(self) -> Session:
        return self._session or db_manager.get_session()

    def insert_state_records(self, dataset_id: str, records: List[Dict[str, Any]]) -> int:
        session = self._get_session()
        try:
            session.query(StateDataModel).filter_by(dataset_id=dataset_id).delete()
            objs = [
                StateDataModel(
                    dataset_id=dataset_id,
                    state=r.get("state") or r.get("State") or "",
                    capital=r.get("capital") or r.get("Capital") or "",
                    state_code=r.get("state_code") or r.get("State_Code")
                )
                for r in records
            ]
            session.bulk_save_objects(objs)
            session.commit()
            return len(objs)
        except Exception:
            session.rollback()
            raise
        finally:
            if not self._session:
                session.close()

    def insert_village_records(self, dataset_id: str, records: List[Dict[str, Any]]) -> int:
        session = self._get_session()
        try:
            session.query(VillageDataModel).filter_by(dataset_id=dataset_id).delete()
            objs = []
            for r in records:
                objs.append(
                    VillageDataModel(
                        dataset_id=dataset_id,
                        state=r.get("state") or r.get("State") or "",
                        capital=r.get("capital") or r.get("Capital") or "",
                        village=r.get("village") or r.get("Village") or r.get("village_name") or "",
                        population=float(r.get("population") or r.get("Population", 0.0) or 0.0),
                        no_of_males=float(r.get("no_of_males") or r.get("No_of_Males", 0.0) or 0.0),
                        no_of_females=float(r.get("no_of_females") or r.get("No_of_Females", 0.0) or 0.0),
                        literacy_rate_percent=float(r.get("literacy_rate_percent") or r.get("Literacy_Rate_Percent", 0.0) or 0.0),
                        area_sq_km=float(r.get("area_sq_km") or r.get("Area_Sq_Km", 0.0) or 0.0),
                        households=float(r.get("households") or r.get("Households", 0.0) or 0.0),
                    )
                )
            session.bulk_save_objects(objs)
            session.commit()
            return len(objs)
        except Exception:
            session.rollback()
            raise
        finally:
            if not self._session:
                session.close()

    save_village_records = insert_village_records
    save_state_records = insert_state_records

    def get_all_villages_dataframe(self) -> pd.DataFrame:
        """Load all village records as Pandas DataFrame."""
        session = self._get_session()
        try:
            rows = session.query(VillageDataModel).all()
            if not rows:
                return pd.DataFrame()
            return pd.DataFrame([
                {
                    "State": r.state,
                    "Capital": r.capital,
                    "Village": r.village,
                    "Population": r.population,
                    "No_of_Males": r.no_of_males,
                    "No_of_Females": r.no_of_females,
                    "Literacy_Rate_Percent": r.literacy_rate_percent,
                    "Area_Sq_Km": r.area_sq_km,
                    "Households": r.households,
                    "dataset_id": r.dataset_id
                }
                for r in rows
            ])
        finally:
            if not self._session:
                session.close()

    def get_all_states(self) -> List[Dict[str, Any]]:
        session = self._get_session()
        try:
            records = session.query(StateDataModel).all()
            return [
                {
                    "state": r.state,
                    "capital": r.capital,
                    "state_code": r.state_code,
                    "dataset_id": r.dataset_id
                }
                for r in records
            ]
        finally:
            if not self._session:
                session.close()

    def get_state_aggregates(self, state_name: str) -> Dict[str, Any]:
        session = self._get_session()
        try:
            res = session.query(
                func.count(VillageDataModel.id).label("village_count"),
                func.sum(VillageDataModel.population).label("total_population"),
                func.avg(VillageDataModel.literacy_rate_percent).label("avg_literacy"),
                func.sum(VillageDataModel.area_sq_km).label("total_area"),
            ).filter(func.lower(VillageDataModel.state) == state_name.strip().lower()).first()

            if res:
                return {
                    "village_count": int(res.village_count or 0),
                    "total_population": float(res.total_population or 0.0),
                    "avg_literacy": float(res.avg_literacy or 0.0),
                    "total_area": float(res.total_area or 0.0),
                }
            return {"village_count": 0, "total_population": 0.0, "avg_literacy": 0.0, "total_area": 0.0}
        finally:
            if not self._session:
                session.close()


class EntityRepository:
    """Repository managing entities and aliases."""

    def __init__(self, session: Optional[Session] = None):
        self._session = session

    def _get_session(self) -> Session:
        return self._session or db_manager.get_session()

    def upsert_entity(
        self,
        entity_type: str,
        name: str,
        entity_id: Optional[str] = None,
        parent_entity_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        dataset_id: Optional[str] = None
    ) -> EntityModel:
        session = self._get_session()
        target_id = entity_id or f"{entity_type.lower()}_{name.strip().lower().replace(' ', '_')}"
        effective_parent = parent_entity_id or parent_id
        try:
            e = session.query(EntityModel).filter_by(entity_id=target_id).first()
            if not e:
                e = EntityModel(
                    entity_id=target_id,
                    entity_type=entity_type,
                    name=name,
                    normalized_name=name.strip().lower(),
                    parent_entity_id=effective_parent,
                    dataset_id=dataset_id
                )
                session.add(e)
            else:
                e.name = name
                e.normalized_name = name.strip().lower()
                e.parent_entity_id = effective_parent
                if dataset_id:
                    e.dataset_id = dataset_id
            session.commit()
            return e
        except Exception:
            session.rollback()
            raise
        finally:
            if not self._session:
                session.close()

    def add_alias(self, entity_id: str, alias: str, alias_type: str = "ABBREVIATION") -> None:
        session = self._get_session()
        try:
            alias_id = f"{entity_id}_{alias.lower().replace(' ', '_')}"
            existing = session.query(EntityAliasModel).filter_by(alias_id=alias_id).first()
            if not existing:
                ea = EntityAliasModel(
                    alias_id=alias_id,
                    entity_id=entity_id,
                    alias=alias,
                    normalized_alias=alias.strip().lower(),
                    alias_type=alias_type
                )
                session.add(ea)
                session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            if not self._session:
                session.close()

    def resolve_alias(self, term: str) -> Optional[str]:
        session = self._get_session()
        try:
            norm = term.strip().lower()
            alias_rec = session.query(EntityAliasModel).filter_by(normalized_alias=norm).first()
            if alias_rec:
                ent = session.query(EntityModel).filter_by(entity_id=alias_rec.entity_id).first()
                if ent:
                    return ent.name
            ent_direct = session.query(EntityModel).filter_by(normalized_name=norm).first()
            if ent_direct:
                return ent_direct.name
            return None
        finally:
            if not self._session:
                session.close()

    def search_entities(self, query: str = "", limit: int = 1000) -> List[Dict[str, Any]]:
        session = self._get_session()
        try:
            q = session.query(EntityModel)
            if query:
                q = q.filter(EntityModel.normalized_name.contains(query.strip().lower()))
            records = q.limit(limit).all()
            return [
                {
                    "entity_id": r.entity_id,
                    "name": r.name,
                    "entity_type": r.entity_type,
                    "parent_id": r.parent_entity_id,
                    "dataset_id": r.dataset_id
                }
                for r in records
            ]
        finally:
            if not self._session:
                session.close()

    def list_aliases(self, limit: int = 1000) -> List[Dict[str, Any]]:
        session = self._get_session()
        try:
            records = session.query(EntityAliasModel, EntityModel).join(
                EntityModel, EntityAliasModel.entity_id == EntityModel.entity_id
            ).limit(limit).all()
            return [
                {
                    "alias_id": ea.alias_id,
                    "alias": ea.alias,
                    "alias_type": ea.alias_type,
                    "canonical_name": em.name,
                    "entity_type": em.entity_type,
                }
                for ea, em in records
            ]
        finally:
            if not self._session:
                session.close()


class RAGDocumentRepository:
    """Repository managing RAG documents and vector search."""

    def __init__(self, session: Optional[Session] = None):
        self._session = session

    def _get_session(self) -> Session:
        return self._session or db_manager.get_session()

    def upsert_document(self, doc_id: str, doc_type: str, content: str, metadata: Optional[Dict[str, Any]] = None, embedding: Optional[List[float]] = None, dataset_id: Optional[str] = None) -> None:
        session = self._get_session()
        try:
            doc = session.query(RAGDocumentModel).filter_by(document_id=doc_id).first()
            emb_str = json.dumps(embedding) if embedding else None
            if not doc:
                doc = RAGDocumentModel(
                    document_id=doc_id,
                    dataset_id=dataset_id,
                    document_type=doc_type,
                    content=content,
                    doc_metadata=metadata or {},
                    embedding=emb_str
                )
                session.add(doc)
            else:
                doc.content = content
                doc.document_type = doc_type
                doc.doc_metadata = metadata or {}
                if emb_str:
                    doc.embedding = emb_str
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            if not self._session:
                session.close()

    def get_all_documents(self, doc_type: Optional[str] = None) -> List[Dict[str, Any]]:
        session = self._get_session()
        try:
            q = session.query(RAGDocumentModel)
            if doc_type:
                q = q.filter_by(document_type=doc_type)
            docs = q.all()
            return [
                {
                    "document_id": d.document_id,
                    "dataset_id": d.dataset_id,
                    "document_type": d.document_type,
                    "content": d.content,
                    "metadata": d.doc_metadata,
                    "embedding": json.loads(d.embedding) if d.embedding else None
                }
                for d in docs
            ]
        finally:
            if not self._session:
                session.close()


class QueryLogRepository:
    """Repository managing query audit logs."""

    @staticmethod
    def log(
        question: str,
        operation: Optional[str],
        result_count: int,
        processing_time: float,
        dataset_name: str = "default",
        session_id: str = "default",
        status: str = "success",
        verification_status: str = "PASS",
        query_plan: Optional[Dict[str, Any]] = None
    ) -> Optional[int]:
        session = db_manager.get_session()
        try:
            log_entry = QueryLogModel(
                timestamp=datetime.now(timezone.utc),
                session_id=session_id,
                dataset_name=dataset_name,
                question=question,
                operation=operation,
                result_count=result_count,
                processing_time=processing_time,
                status=status,
                verification_status=verification_status,
                query_plan=query_plan
            )
            session.add(log_entry)
            session.commit()
            return log_entry.id
        except Exception as ex:
            session.rollback()
            logger.warning(f"QueryLogRepository log failed: {ex}")
            return None
        finally:
            session.close()


dataset_repo = DatasetRepository()
state_village_repo = StateVillageRepository()
entity_repo = EntityRepository()
rag_doc_repo = RAGDocumentRepository()
query_log_repo = QueryLogRepository()
