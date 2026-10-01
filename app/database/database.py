"""SQLite database for dataset registry, query audit trail, and conversation metadata."""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from app.config import settings
from app.utils.logger import logger


def get_db_path() -> Path:
    db_path = Path(settings.DATABASE_PATH)
    if not db_path.is_absolute():
        db_path = Path(__file__).resolve().parent.parent.parent / db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


def init_db() -> None:
    """Create database tables and auto-migrate columns if needed."""
    path = get_db_path()
    try:
        with sqlite3.connect(path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS query_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    dataset_name TEXT DEFAULT 'default',
                    session_id TEXT DEFAULT 'default',
                    question TEXT NOT NULL,
                    operation TEXT,
                    result_count INTEGER DEFAULT 0,
                    processing_time REAL DEFAULT 0.0,
                    status TEXT DEFAULT 'success'
                )
                """
            )
            # Safe migration for existing table
            cursor.execute("PRAGMA table_info(query_history)")
            existing_cols = [col[1] for col in cursor.fetchall()]
            if "dataset_name" not in existing_cols:
                cursor.execute("ALTER TABLE query_history ADD COLUMN dataset_name TEXT DEFAULT 'default'")
            if "session_id" not in existing_cols:
                cursor.execute("ALTER TABLE query_history ADD COLUMN session_id TEXT DEFAULT 'default'")
            if "operation" not in existing_cols:
                cursor.execute("ALTER TABLE query_history ADD COLUMN operation TEXT DEFAULT 'FILTER'")

            conn.commit()
        logger.info(f"SQLite database initialized at {path}")
    except Exception as e:
        logger.error(f"Error initializing database at {path}: {e}")


init_db()


def log_query(
    dataset_name: str,
    question: str,
    operation: str,
    result_count: int,
    processing_time: float,
    status: str = "success",
    session_id: Optional[str] = None
) -> Optional[int]:
    """Audit log every question and execution timing in SQLite."""
    init_db()
    path = get_db_path()
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    try:
        with sqlite3.connect(path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO query_history (timestamp, dataset_name, session_id, question, operation, result_count, processing_time, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (now_str, dataset_name, session_id or "default", question, operation, result_count, processing_time, status)
            )
            conn.commit()
            return cursor.lastrowid
    except Exception as e:
        logger.error(f"Failed to log query to database: {e}")
        return None


def get_query_history(limit: int = 20) -> List[Dict[str, Any]]:
    """Return recent query audit logs."""
    init_db()
    path = get_db_path()
    try:
        with sqlite3.connect(path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, timestamp, dataset_name, question, operation, result_count, processing_time, status
                FROM query_history
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,)
            )
            return [dict(r) for r in cursor.fetchall()]
    except Exception as e:
        logger.error(f"Failed to fetch query history: {e}")
        return []
