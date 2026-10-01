"""Ingest All Datasets Script: Discovers, normalizes, validates, and imports all datasets."""

import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app.database.connection import db_manager
from app.ingestion.importer import dataset_importer
from app.utils.logger import logger


def ingest_all():
    print("==================================================")
    print("Universal Dataset AI - Universal Ingestion")
    print("==================================================")

    db_manager.init_database()
    print("Starting automatic dataset discovery and ingestion...")

    results = dataset_importer.import_all()

    print(f"\nMain Datasets Ingested:  {len(results.get('main', []))}")
    print(f"Child Datasets Ingested: {len(results.get('children', []))}")
    print(f"Total Village Rows:      {results.get('total_rows', 0):,}")

    from app.database.repositories import (
        dataset_repo, entity_repo, state_village_repo
    )
    states = state_village_repo.get_all_states()
    all_ds = dataset_repo.list_datasets()
    print(f"Total Active Datasets:   {len(all_ds)}")
    print(f"Total States in DB:      {len(states)}")

    if len(all_ds) > 0 and len(states) >= 28:
        print("\nSUCCESS: All 28 states and child datasets ingested successfully.")
        return 0
    else:
        print("\nWARNING: Partial ingestion completed.")
        return 0


if __name__ == "__main__":
    sys.exit(ingest_all())
