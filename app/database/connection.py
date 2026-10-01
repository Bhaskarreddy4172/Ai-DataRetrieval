"""Database Connection Manager: PostgreSQL + pgvector with seamless SQLite fallback."""

import logging
import threading
from typing import Any, Dict, Generator, Optional
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from app.config import settings
from app.utils.logger import logger
from app.database.models import Base

logger = logging.getLogger("ai_dataset_retrieval")


class DatabaseManager:
    """Singleton database manager managing connection pooling, dialect detection, and sessions."""

    _instance: Optional["DatabaseManager"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker] = None
        self._is_postgres: bool = False
        self._has_pgvector: bool = False
        self._active_url: str = ""
        self._initialized: bool = False
        self._initialize()

    def _initialize(self) -> None:
        """Initialize engine: try PostgreSQL first, fall back to SQLite if unreachable."""
        with self._lock:
            pg_url = settings.DATABASE_URL
            engine = None

            # 1. Attempt PostgreSQL
            if pg_url and pg_url.startswith("postgresql"):
                try:
                    candidate_engine = create_engine(
                        pg_url,
                        pool_size=10,
                        max_overflow=20,
                        pool_pre_ping=True,
                        connect_args={"connect_timeout": 3}
                    )
                    with candidate_engine.connect() as conn:
                        conn.execute(text("SELECT 1"))
                        # Try enabling pgvector
                        try:
                            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                            conn.commit()
                            self._has_pgvector = True
                        except Exception as pge:
                            logger.info(f"pgvector extension check note: {pge}")
                    engine = candidate_engine
                    self._is_postgres = True
                    self._active_url = pg_url
                    logger.info("Database: Connected to PostgreSQL successfully.")
                except Exception as ex:
                    logger.info(f"Database: PostgreSQL unreachable ({ex}); falling back to SQLite.")
                    engine = None

            # 2. Fallback to SQLite
            if engine is None:
                sqlite_url = settings.FALLBACK_SQLITE_URL
                engine = create_engine(
                    sqlite_url,
                    connect_args={"check_same_thread": False}
                )
                self._is_postgres = False
                self._has_pgvector = False
                self._active_url = sqlite_url
                logger.info(f"Database: Using SQLite database at {sqlite_url}.")

            self._engine = engine
            self._session_factory = sessionmaker(autocommit=False, autoflush=False, bind=self._engine)
            self._initialized = True

    def get_engine(self) -> Engine:
        if self._engine is None:
            self._initialize()
        return self._engine  # type: ignore

    def get_session(self) -> Session:
        if self._session_factory is None:
            self._initialize()
        return self._session_factory()  # type: ignore

    def init_database(self) -> None:
        """Create all database tables idempotently."""
        engine = self.get_engine()
        try:
            # Check for legacy datasets table schema (e.g. from older sqlite versions)
            with engine.connect() as conn:
                res = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='datasets'")).fetchone() if not self._is_postgres else None
                if res:
                    cols = [row[1] for row in conn.execute(text("PRAGMA table_info(datasets)")).fetchall()]
                    if "dataset_id" not in cols:
                        logger.info("Migrating legacy datasets table schema...")
                        conn.execute(text("DROP TABLE datasets"))
                        conn.commit()

            Base.metadata.create_all(bind=engine)
            logger.info("Database: All relational tables verified/created successfully.")
        except Exception as ex:
            logger.error(f"Database: Table creation error: {ex}")

    def check_health(self) -> Dict[str, Any]:
        """Perform database health verification."""
        try:
            engine = self.get_engine()
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return {
                "status": "READY",
                "backend": "PostgreSQL" if self._is_postgres else "SQLite",
                "pgvector": "ENABLED" if self._has_pgvector else "DISABLED (Using hybrid vector fallback)",
                "database_url": self._active_url.split("@")[-1] if "@" in self._active_url else self._active_url,
                "healthy": True
            }
        except Exception as ex:
            return {
                "status": "UNAVAILABLE",
                "backend": "Unknown",
                "error": str(ex),
                "healthy": False
            }


db_manager = DatabaseManager()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for database session lifecycle."""
    session = db_manager.get_session()
    try:
        yield session
    finally:
        session.close()


def init_db() -> None:
    """Initialize database tables and schemas."""
    db_manager.init_database()
