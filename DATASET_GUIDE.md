
# Dataset Ingestion & Profiling Guide

## Overview

The Universal AI Dataset System dynamically accepts and indexes structured tabular datasets without requiring hardcoded column names, prior domain knowledge, or custom schemas.

---

## Supported File Formats

| Format                           | Extensions          | Notes                                                |
| -------------------------------- | ------------------- | ---------------------------------------------------- |
| **Comma-Separated Values** | `.csv`            | UTF-8 / ASCII encoding supported automatically       |
| **Excel Workbook**         | `.xlsx`, `.xls` | Multi-sheet navigation and sheet switching supported |
| **JSON Records**           | `.json`           | Array of objects or records format                   |

---

## Automatic Schema Discovery & Profiling

Upon upload, the system automatically performs:

1. **Row & Column Sanitation**: Normalizes column header names and trims whitespace.
2. **Data Type Inference**: Categorizes fields into:
   - `TEXT` / `ENTITY` (Names, IDs, Cities, Departments, Categories)
   - `NUMBER` (Salaries, Amounts, Scores, Quantities, Prices)
   - `DATE` (Joining Date, Order Date, Timestamp)
3. **Internal Row ID Binding**: Attaches `_internal_row_id` to every record to guarantee row identity during complex multi-step filtering and joins.
4. **Dynamic Spell Vocabulary**: Indexes unique text values and column titles into the fuzzy/phonetic spell checker engine (Levenshtein, Metaphone, RapidFuzz).
5. **Relationship Detection**: Identifies candidate keys and correlation patterns across columns.

---

## Multi-Dataset Isolation

- Every uploaded dataset receives a unique `dataset_id` and MD5 `content_hash`.
- Uploading or switching datasets automatically:
  - Clears previous session conversation history.
  - Invalidates active query caches.
  - Re-indexes vocabulary for the new dataset.
  - Prevents cross-dataset fact leakage.
