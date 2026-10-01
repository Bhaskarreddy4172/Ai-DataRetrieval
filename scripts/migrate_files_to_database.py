"""Alias script for migrate_to_database.py."""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from scripts.migrate_to_database import migrate_to_database

if __name__ == "__main__":
    sys.exit(migrate_to_database())
