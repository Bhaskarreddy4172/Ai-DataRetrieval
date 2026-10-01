"""Database Backup Script: Exports database snapshot for backup and persistence."""

import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.config import settings
from app.database.connection import db_manager
from app.utils.logger import logger


def backup_database():
    print("==================================================")
    print("Universal Dataset AI - Database Backup Utility")
    print("==================================================")

    backup_dir = BASE_DIR / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    health = db_manager.check_health()
    backend = health.get("backend")

    if backend == "PostgreSQL":
        backup_file = backup_dir / f"pg_backup_{timestamp}.sql"
        print(f"Backing up PostgreSQL database to {backup_file.name}...")
        db_url = settings.DATABASE_URL
        # Try pg_dump if available
        try:
            cmd = f"pg_dump {db_url} -f {backup_file}"
            res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if res.returncode == 0:
                print(f"SUCCESS: PostgreSQL backup saved to {backup_file}")
                return 0
            else:
                logger.warning(f"pg_dump output: {res.stderr}")
        except Exception as ex:
            logger.warning(f"pg_dump failed: {ex}")

    # Fallback or SQLite backup
    sqlite_db = BASE_DIR / "app.db"
    if sqlite_db.exists():
        backup_file = backup_dir / f"sqlite_backup_{timestamp}.db"
        shutil.copy2(sqlite_db, backup_file)
        print(f"SUCCESS: SQLite database backup saved to {backup_file}")
        return 0

    print("WARNING: No database file found to backup.")
    return 1


if __name__ == "__main__":
    sys.exit(backup_database())
