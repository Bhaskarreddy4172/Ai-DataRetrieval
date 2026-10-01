"""Ingestion module initialization."""

from app.ingestion.schema_detector import schema_detector, SchemaDetector
from app.ingestion.column_mapper import column_mapper, ColumnMapper
from app.ingestion.normalizer import normalizer, DataNormalizer
from app.ingestion.validator import ingestion_validator, IngestionValidator
from app.ingestion.csv_loader import csv_loader, CSVLoader
from app.ingestion.excel_loader import excel_loader, ExcelLoader
from app.ingestion.importer import dataset_importer, DatasetImporter

__all__ = [
    "schema_detector",
    "SchemaDetector",
    "column_mapper",
    "ColumnMapper",
    "normalizer",
    "DataNormalizer",
    "ingestion_validator",
    "IngestionValidator",
    "csv_loader",
    "CSVLoader",
    "excel_loader",
    "ExcelLoader",
    "dataset_importer",
    "DatasetImporter",
]
