"""SQLAlchemy Database Models for Universal Dataset AI Platform.

Supports both PostgreSQL + pgvector and SQLite fallback.
Tables:
- datasets
- dataset_columns
- dataset_relationships
- entities
- entity_aliases
- state_data
- village_data
- rag_documents
- conversation_sessions
- conversation_messages
- query_logs
- validation_results
"""

from datetime import datetime, timezone
from typing import Any, Dict, Optional
from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey, Integer,
    String, Text, JSON, Index, func
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class DatasetModel(Base):
    """Catalog entry for discovered, validated, and ingested datasets."""
    __tablename__ = "datasets"

    dataset_id = Column(String(100), primary_key=True)
    dataset_name = Column(String(255), nullable=False)
    file_name = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_type = Column(String(20), nullable=False)  # CSV, XLSX, XLS, JSON
    dataset_type = Column(String(50), nullable=False, default="CHILD")  # MAIN, CHILD, STANDALONE
    parent_dataset_id = Column(String(100), nullable=True)
    parent_entity = Column(String(100), nullable=True)  # e.g., "State"
    child_entity = Column(String(100), nullable=True)   # e.g., "Village"
    row_count = Column(Integer, default=0)
    column_count = Column(Integer, default=0)
    content_hash = Column(String(64), nullable=False)
    version = Column(Integer, default=1)
    status = Column(String(50), nullable=False, default="DISCOVERED")  # DISCOVERED, VALIDATING, IMPORTING, INDEXING, READY, FAILED, STALE
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    columns = relationship("DatasetColumnModel", back_populates="dataset", cascade="all, delete-orphan")


class DatasetColumnModel(Base):
    """Metadata and semantic classifications for individual dataset columns."""
    __tablename__ = "dataset_columns"

    column_id = Column(String(150), primary_key=True)
    dataset_id = Column(String(100), ForeignKey("datasets.dataset_id"), nullable=False)
    original_name = Column(String(255), nullable=False)
    normalized_name = Column(String(255), nullable=False)
    semantic_type = Column(String(100), nullable=False)  # POPULATION, AREA, MALE_COUNT, FEMALE_COUNT, LITERACY_RATE, HOUSEHOLDS, STATE, CAPITAL, VILLAGE, etc.
    data_type = Column(String(50), nullable=False)  # integer, float, string, date, boolean
    is_numeric = Column(Boolean, default=False)
    is_filterable = Column(Boolean, default=True)
    is_aggregatable = Column(Boolean, default=False)

    dataset = relationship("DatasetModel", back_populates="columns")


class DatasetRelationshipModel(Base):
    """Structural parent-child or relational mappings between registered datasets."""
    __tablename__ = "dataset_relationships"

    relationship_id = Column(String(150), primary_key=True)
    parent_dataset_id = Column(String(100), nullable=False)
    child_dataset_id = Column(String(100), nullable=False)
    parent_column = Column(String(255), nullable=False)
    child_column = Column(String(255), nullable=False)
    relationship_type = Column(String(50), nullable=False, default="ONE_TO_MANY")


class EntityModel(Base):
    """Normalized entities (States, Capitals, Villages) discovered across datasets."""
    __tablename__ = "entities"

    entity_id = Column(String(150), primary_key=True)
    entity_type = Column(String(50), nullable=False)  # State, Capital, Village
    name = Column(String(255), nullable=False)
    normalized_name = Column(String(255), nullable=False, index=True)
    parent_entity_id = Column(String(150), nullable=True)
    dataset_id = Column(String(100), nullable=True)

    aliases = relationship("EntityAliasModel", back_populates="entity", cascade="all, delete-orphan")


class EntityAliasModel(Base):
    """Aliases, abbreviations, Hindi/Telugu variations, and typo corrections for entities."""
    __tablename__ = "entity_aliases"

    alias_id = Column(String(150), primary_key=True)
    entity_id = Column(String(150), ForeignKey("entities.entity_id"), nullable=False)
    alias = Column(String(255), nullable=False)
    normalized_alias = Column(String(255), nullable=False, index=True)
    alias_type = Column(String(50), nullable=False, default="ABBREVIATION")  # ABBREVIATION, TYPO, HINGLISH, TELUGU_ENGLISH, CODE

    entity = relationship("EntityModel", back_populates="aliases")


class StateDataModel(Base):
    """Main state-level dataset records."""
    __tablename__ = "state_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(String(100), nullable=False, index=True)
    state = Column(String(100), nullable=False, index=True)
    capital = Column(String(100), nullable=False, index=True)
    state_code = Column(String(10), nullable=True, index=True)


class VillageDataModel(Base):
    """Child village-level dataset records across all 28 registered states."""
    __tablename__ = "village_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(String(100), nullable=False, index=True)
    state_id = Column(Integer, nullable=True)
    state = Column(String(100), nullable=False, index=True)
    capital = Column(String(100), nullable=False, index=True)
    village = Column(String(150), nullable=False, index=True)
    population = Column(Float, nullable=False, default=0.0, index=True)
    no_of_males = Column(Float, nullable=False, default=0.0)
    no_of_females = Column(Float, nullable=False, default=0.0)
    literacy_rate_percent = Column(Float, nullable=False, default=0.0)
    area_sq_km = Column(Float, nullable=False, default=0.0, index=True)
    households = Column(Float, nullable=False, default=0.0)


class RAGDocumentModel(Base):
    """Structured RAG documentation for schema, column semantics, entities, and context."""
    __tablename__ = "rag_documents"

    document_id = Column(String(150), primary_key=True)
    dataset_id = Column(String(100), nullable=True, index=True)
    document_type = Column(String(50), nullable=False, index=True)  # DATASET_SCHEMA, COLUMN_DESCRIPTION, ENTITY, ENTITY_ALIAS, RELATIONSHIP, ROW_CONTEXT, DATASET_SUMMARY
    content = Column(Text, nullable=False)
    doc_metadata = Column(JSON, nullable=True)
    embedding = Column(Text, nullable=True)  # JSON-encoded array on SQLite, pgvector on PostgreSQL when enabled


class ConversationSessionModel(Base):
    """Conversation dialogue session tracking."""
    __tablename__ = "conversation_sessions"

    session_id = Column(String(100), primary_key=True)
    title = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    messages = relationship("ConversationMessageModel", back_populates="session", cascade="all, delete-orphan")


class ConversationMessageModel(Base):
    """Messages and turns within conversation dialogue sessions."""
    __tablename__ = "conversation_messages"

    message_id = Column(String(150), primary_key=True)
    session_id = Column(String(100), ForeignKey("conversation_sessions.session_id"), nullable=False, index=True)
    role = Column(String(50), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    metadata_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    session = relationship("ConversationSessionModel", back_populates="messages")


class QueryLogModel(Base):
    """Persistent query audit trail, execution performance, and verification status."""
    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    session_id = Column(String(100), nullable=False, default="default", index=True)
    dataset_name = Column(String(255), nullable=False, default="default")
    question = Column(Text, nullable=False)
    operation = Column(String(50), nullable=True)
    result_count = Column(Integer, default=0)
    processing_time = Column(Float, default=0.0)
    status = Column(String(50), default="success")
    verification_status = Column(String(50), default="PASS")
    query_plan = Column(JSON, nullable=True)


class ValidationResultModel(Base):
    """Dataset and schema integrity validation records."""
    __tablename__ = "validation_results"

    validation_id = Column(String(150), primary_key=True)
    dataset_id = Column(String(100), nullable=False, index=True)
    check_type = Column(String(100), nullable=False)
    status = Column(String(50), nullable=False)  # PASS, FAIL, WARNING
    details = Column(Text, nullable=False)
    validated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class QueryPlanModel(Base):
    """Structured JSON query plans for audit, debugging, and deterministic execution."""
    __tablename__ = "query_plans"

    plan_id = Column(String(150), primary_key=True)
    session_id = Column(String(100), nullable=False, default="default", index=True)
    question = Column(Text, nullable=False)
    structured_plan = Column(JSON, nullable=False)
    execution_type = Column(String(50), nullable=False, default="SQL")  # SQL, DUCKDB, RAG, BOOLEAN
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class QueryResultModel(Base):
    """Execution results and verification cache."""
    __tablename__ = "query_results"

    result_id = Column(String(150), primary_key=True)
    plan_id = Column(String(150), ForeignKey("query_plans.plan_id"), nullable=True, index=True)
    data = Column(JSON, nullable=True)
    result_count = Column(Integer, default=0)
    execution_time = Column(Float, default=0.0)
    verified = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class DatabaseMigrationModel(Base):
    """Database schema migration version tracking."""
    __tablename__ = "database_migrations"

    migration_id = Column(String(150), primary_key=True)
    version = Column(Integer, nullable=False, unique=True)
    description = Column(String(255), nullable=False)
    applied_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    status = Column(String(50), nullable=False, default="APPLIED")


class SystemHealthModel(Base):
    """System health audit logs."""
    __tablename__ = "system_health"

    id = Column(Integer, primary_key=True, autoincrement=True)
    component = Column(String(100), nullable=False, index=True)
    status = Column(String(50), nullable=False)
    details = Column(JSON, nullable=True)
    checked_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class EntityValueModel(Base):
    """Specific entity attribute values for fast relational and fuzzy matching."""
    __tablename__ = "entity_values"

    value_id = Column(String(150), primary_key=True)
    entity_id = Column(String(150), ForeignKey("entities.entity_id"), nullable=False, index=True)
    column_name = Column(String(100), nullable=False)
    raw_value = Column(String(255), nullable=False)
    normalized_value = Column(String(255), nullable=False, index=True)


# Indices for high-performance retrieval and filter execution
Index("ix_village_state_pop", VillageDataModel.state, VillageDataModel.population)
Index("ix_village_state_area", VillageDataModel.state, VillageDataModel.area_sq_km)
Index("ix_dataset_status", DatasetModel.status)
Index("ix_entity_alias_normalized", EntityAliasModel.normalized_alias)
Index("ix_entity_val_norm", EntityValueModel.normalized_value)

