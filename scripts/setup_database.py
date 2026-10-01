"""Database Setup Script: Initializes schema, tables, and extensions."""

import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.database.connection import db_manager
from app.utils.logger import logger


def setup_database():
    print("==================================================")
    print("Universal Dataset AI - Database Initialization")
    print("==================================================")

    logger.info("Initializing database connection...")
    db_manager.init_database()

    health = db_manager.check_health()
    print(f"Database Backend: {health.get('backend')}")
    print(f"pgvector Status:  {health.get('pgvector')}")
    print(f"Database URL:     {health.get('database_url')}")
    print(f"Health Status:    {health.get('status')}")

    if health.get("healthy"):
        print("\nSUCCESS: Database initialized and ready for ingestion.")
        return 0
    else:
        print("\nERROR: Database health check failed.")
        return 1


if __name__ == "__main__":
    sys.exit(setup_database())
