"""Database Restore Script: Restores database from a snapshot."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.config import settings
from app.database.connection import db_manager
from app.utils.logger import logger


def restore_database(backup_path: str = ""):
    print("==================================================")
    print("Universal Dataset AI - Database Restore Utility")
    print("==================================================")

    backup_dir = BASE_DIR / "backups"
    if not backup_dir.exists():
        print("ERROR: No backups directory found.")
        return 1

    if backup_path:
        target = Path(backup_path)
    else:
        # Find newest backup
        backups = sorted(backup_dir.glob("*.*"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not backups:
            print("ERROR: No backup files found in backups/ directory.")
            return 1
        target = backups[0]

    print(f"Restoring database from {target.name}...")

    if target.suffix == ".sql":
        db_url = settings.DATABASE_URL
        try:
            cmd = f"psql {db_url} -f {target}"
            res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if res.returncode == 0:
                print("SUCCESS: PostgreSQL database restored successfully.")
                return 0
            else:
                logger.error(f"psql error: {res.stderr}")
                return 1
        except Exception as ex:
            logger.error(f"PostgreSQL restore failed: {ex}")
            return 1
    elif target.suffix == ".db":
        dest_db = BASE_DIR / "app.db"
        shutil.copy2(target, dest_db)
        print("SUCCESS: SQLite database restored successfully.")
        return 0
    else:
        print(f"ERROR: Unsupported backup file format: {target.suffix}")
        return 1


if __name__ == "__main__":
    b_path = sys.argv[1] if len(sys.argv) > 1 else ""
    sys.exit(restore_database(b_path))
