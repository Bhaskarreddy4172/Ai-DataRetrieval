# System Architecture Specification

## Executive Overview

The **Universal AI Dataset Chatbot** is a production-grade, conversational natural language dataset question-answering platform designed for zero-hallucination factual analysis over structured tabular datasets (CSV, Excel `.xlsx`/`.xls`, JSON, and multi-sheet workbooks).

The system decouples **language understanding & planning** (handled by local LLMs via Ollama) from **factual computation & data retrieval** (handled deterministically by Pandas and DuckDB).

---

## High-Level Architecture Diagram

```text
                                  USER QUESTION
                                        │
                                        ▼
                            ┌───────────────────────┐
                            │  FastAPI Router       │
                            └───────────┬───────────┘
                                        ▼
                            ┌───────────────────────┐
                            │ Language Normalizer   │ (Typo, Phonetic, Hinglish, Abbreviations)
                            └───────────┬───────────┘
                                        ▼
                            ┌───────────────────────┐
                            │ Context & Memory      │ (Pronoun resolution, Turn history)
                            └───────────┬───────────┘
                                        ▼
                            ┌───────────────────────┐
                            │ Intent & Router       │ (Dataset vs GK vs Hybrid vs Ambiguous)
                            └───────────┬───────────┘
                                        ▼
                            ┌───────────────────────┐
                            │ Ollama LLM / Tool     │ (Qwen / Llama / Deterministic Fallback)
                            └───────────┬───────────┘
                                        ▼
                            ┌───────────────────────┐
                            │ Master Orchestrator   │ (State Machine: PLAN -> TOOL -> VERIFY)
                            └───────────┬───────────┘
                                        ▼
                            ┌───────────────────────┐
                            │ Safe Tool Registry    │ (Pandas & DuckDB Tool Execution)
                            └───────────┬───────────┘
                                        ▼
                            ┌───────────────────────┐
                            │ Hallucination Firewall│ (0.0% Hallucination Verification)
                            └───────────┬───────────┘
                                        ▼
                            ┌───────────────────────┐
                            │ Natural Language Ans. │
                            └───────────────────────┘
```

---

## Core System Components

### 1. Request Lifecycle & Orchestrator (`app/agent/`)
- **State Machine**: Controls 13 request states (`RECEIVED` → `UNDERSTANDING` → `CONTEXT_RESOLUTION` → `ROUTING` → `PLANNING` → `TOOL_SELECTION` → `EXECUTING` → `OBSERVING` → `REPLANNING` → `VERIFICATION` → `ANSWER_GENERATION` → `COMPLETED`).
- **Context Resolver**: Resolves relative references (*"her"*, *"that city"*, *"the second employee"*) using conversation state.
- **Ambiguity Guard**: Triggers explicit clarification choices when multiple entity candidates exist.

### 2. Dataset Ingestion & Profiling (`app/dataset/`)
- **Schema Intelligence**: Detects entity columns, numeric metrics, date fields, categories, candidate keys, and multi-sheet Excel structures.
- **Dynamic Spell Checker**: Builds custom vocabulary from active dataset values using Levenshtein distance, RapidFuzz, Soundex, and Metaphone phonetic algorithms.
- **Dataset Isolation**: Generates unique `dataset_id` and `content_hash` to ensure zero cross-dataset context leakage.

### 3. Safe Tool Architecture (`app/tools/`)
- Controlled registry containing 11 deterministic tool tools (`search_dataset`, `lookup`, `filter_dataset`, `aggregate_dataset`, `sort_dataset`, `top_n`, `bottom_n`, `group_by`, `compare`, `statistics`, `schema_inspection`).
- **Zero Arbitrary Execution**: Prohibits raw SQL or Python code execution from LLMs.

### 4. Zero Hallucination Firewall (`app/verification/`)
- Validates LLM outputs against factual DataFrame tool execution results before user delivery.
- Rejects fabricated values or fallback to factual grounded templates.

### 5. Configurable Ollama LLM Provider (`app/llm/` & `app/ai/`)
- Health tracking enums (`READY`, `UNAVAILABLE`, `MODEL_NOT_FOUND`, `TIMEOUT`, `ERROR`).
- Retries with exponential backoff (`OLLAMA_MAX_RETRIES`).
- Fallback hierarchy: `Primary Ollama Model` → `Plan Repair` → `Deterministic Safe Fallback` → `Clarification` → `Safe Failure`.

---

## Security & Isolation Architecture

1. **Upload Sanitation**: Filename sanitization, file size limits (50MB), extension validation.
2. **Context Isolation**: Session-isolated conversation memory and query cache invalidation on dataset switch.
3. **Model Control Scoping**: Model is strictly constrained to tool parameter selection.

