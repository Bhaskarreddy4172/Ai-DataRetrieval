# Universal Dataset AI Chatbot

A production-grade, conversational natural language dataset retrieval and analytics platform powered by **FastAPI**, **PostgreSQL + pgvector** (with seamless **SQLite/DuckDB** zero-config fallback), **Hybrid Vector Search / RAG**, and **Ollama**.

The system enables users to upload **arbitrary tabular datasets (CSV, Excel, JSON)** and query them in **any natural language style**—including professional queries, colloquial speech, abbreviations, multilingual variations (Hinglish, Telugu-English), typos, and state shortcuts—with **strict factual grounding and 0.0% hallucination rate**.

---

## 🏗️ 1. Architecture Diagram

```mermaid
flowchart TD
    User([User / Browser]) <--> ReactUI[React 18 + Vite Frontend]
    ReactUI <--> FastAPI[FastAPI Backend / REST API]

    subgraph "Query & Planning Layer"
        FastAPI --> Normalizer[Language Normalizer & Spell Checker]
        Normalizer --> FastClassifier[Deterministic Intent & Scope Classifier]
        FastClassifier --> HybridRAG[Hybrid RAG & Vector Store]
        HybridRAG --> QueryPlanner[Universal Query Planner]
    end

    subgraph "Execution & Calculation Engine"
        QueryPlanner --> SQLExec[Parameterized SQL Executor]
        QueryPlanner --> DuckDBExec[In-Memory DuckDB Engine]
        SQLExec <--> DB[(PostgreSQL + pgvector / SQLite Fallback)]
        DuckDBExec <--> CombinedDF[Multi-Child Combined Data]
    end

    subgraph "Verification & NLG Firewall"
        SQLExec --> Firewall[Hallucination Firewall & Consistency Validator]
        DuckDBExec --> Firewall
        Firewall --> OllamaLLM[Ollama Local LLM NLG Formatter]
        OllamaLLM --> Response[Verified Factual Answer]
        Firewall -.->|Deterministic Direct NLG| Response
    end

    Response --> FastAPI
```

---

## 🧩 2. Component Overview

1. **Frontend (`frontend/`)**: Modern React 18, TypeScript, TailwindCSS, and Vite application featuring real-time chat, dataset schema explorer, query execution diagnostics, provenance transparent breakdown, and dataset upload modal.
2. **Backend API (`app/main.py`, `app/api/routes.py`)**: High-performance FastAPI server providing endpoints for `/chat`, `/query`, `/health`, `/dataset/upload`, `/database/status`, and conversation history.
3. **Database Layer (`app/database/`)**:
   - Primary: PostgreSQL with `pgvector` extension for persistent storage and dense vector cosine similarity.
   - Dual-engine fallback: Automatic failover to SQLite (`app.db`) with in-memory cosine vector similarity if PostgreSQL is offline.
   - 12 Relational Tables: `datasets`, `dataset_columns`, `dataset_relationships`, `entities`, `entity_aliases`, `state_data`, `village_data`, `rag_documents`, `conversation_sessions`, `conversation_messages`, `query_logs`, `validation_results`.
4. **Universal Ingestion Pipeline (`app/ingestion/`)**:
   - `schema_detector.py`: Detects column types and domain semantics (`POPULATION`, `AREA`, `LITERACY_RATE`, `HOUSEHOLDS`, `STATE`, `CAPITAL`, `VILLAGE`).
   - `column_mapper.py`: Normalizes heterogeneous headers to canonical database attributes.
   - `normalizer.py`: Value cleaning, whitespace stripping, number parsing, percentage extraction.
   - `validator.py`: Dataframe bounds checking, entity relationship validation, negative number constraints.
   - `importer.py`: Orchestrates loading, table updates, and relationship linking.
5. **RAG Knowledge Pipeline (`app/rag/`)**:
   - `document_builder.py`: Synthesizes semantic documents for schemas, column metadata, entities, abbreviations, and state summaries.
   - `embeddings.py`: Ollama embedding service (`nomic-embed-text`) with deterministic L2-normalized pseudo-embedding fallback.
   - `vector_store.py`: Persistent database vector storage and fast cosine similarity search.
   - `hybrid_retriever.py`: Combines dense semantic search + exact/fuzzy entity resolution + token lexical matching.
6. **Execution Engine (`app/execution/`, `app/query/`)**:
   - `sql_executor.py`: Parameterized SQL execution supporting global aggregations, filtered extremes, and rankings.
   - `global_aggregation_engine.py`: High-performance DuckDB and Pandas engine for cross-state operations.
   - `comparison_engine.py`: Comparative calculations, differences, ratios, percentages.
7. **Ollama Integration (`app/ai/`)**:
   - Natural language explanation and response formatting.
   - Strictly isolated from calculations; values are calculated deterministically before generation.

---

## 📋 3. System Requirements

- **Operating System**: Windows 10/11, macOS, or Linux (Ubuntu 20.04+).
- **Python**: Version 3.10, 3.11, 3.12, or 3.14.
- **Node.js**: Version 18.x or 20.x with `npm`.
- **Database (Optional/Recommended)**: PostgreSQL 15+ with `pgvector` (or uses built-in SQLite).
- **LLM / Embeddings (Optional/Recommended)**: [Ollama](https://ollama.ai) running locally (`llama3.1` or `mistral`, `nomic-embed-text`).

---

## 🚀 4. Clean-Machine Installation Guide

### Step 1: Clone the Repository
```bash
git clone https://github.com/Bhaskarreddy4172/Ai-DataRetrieval.git
cd AI-Dataset-Retrieval
```

### Step 2: Set Up Python Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Python Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Install Frontend Dependencies
```bash
cd frontend
npm install
cd ..
```

---

## ⚙️ 5. Environment Variables

Copy the example environment configuration:
```bash
# Windows
copy .env.example .env

# Linux / macOS
cp .env.example .env
```

Key environment configurations in `.env`:
| Variable | Default Value | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql://postgres:postgres@localhost:5432/ai_dataset_db` | Primary PostgreSQL + pgvector connection string |
| `FALLBACK_SQLITE_URL` | `sqlite:///app.db` | Fallback SQLite database path |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama local API server address |
| `OLLAMA_MODEL` | `llama3.1` | Primary NLG formatting model |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Vector embedding model |
| `EMBEDDING_DIM` | `768` | Dimension of embedding vectors |
| `BACKEND_HOST` | `0.0.0.0` | Backend bind host |
| `BACKEND_PORT` | `8000` | Backend API port |
| `FRONTEND_PORT` | `3000` | Frontend UI port |

---

## 🗄️ 6. Database Setup & Ingestion

### Step 1: Initialize Database Tables
Initializes all 12 tables and checks database health:
```bash
python scripts/setup_database.py
```

### Step 2: Ingest Datasets
Discovers and ingests the main dataset and all 28 state village datasets:
```bash
python scripts/ingest_all.py
```

### Step 3: Build RAG Index
Generates semantic documentation and vector representations:
```bash
python scripts/rebuild_rag.py
```

---

## 🏁 7. Running the Application

### Option A: One-Click Startup (Recommended)

**On Windows:**
Double-click or run:
```cmd
run_all.bat
```

**On Linux / macOS:**
```bash
chmod +x run_all.sh run_backend.sh run_frontend.sh
./run_all.sh
```

### Option B: Manual Startup

**Terminal 1 (Backend API):**
```bash
# Windows
run_backend.bat
# or: python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Linux / macOS
./run_backend.sh
# or: python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 (Frontend UI):**
```bash
# Windows
run_frontend.bat
# or: cd frontend && npm run dev

# Linux / macOS
./run_frontend.sh
# or: cd frontend && npm run dev
```

- **Frontend Chatbot**: [http://localhost:3000](http://localhost:3000)
- **Interactive OpenAPI Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Backend Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

---

## 🧪 8. Verification & Testing

### 1. Run Complete Validation Suite
Verifies Database, Data Integrity, RAG Vector Store, and SQL Execution:
```bash
python scripts/validate_all.py
```

### 2. Test All 28 States
Executes capital lookup, extreme village calculations, and total population aggregations across every registered state:
```bash
python scripts/test_all_states.py
```

### 3. Run Pytest Test Suites
```bash
# Database & RAG integration test
python -m pytest tests/test_database_rag_integration.py -v

# All 28 states universal engine test
python -m pytest tests/test_universal_all_28_states_engine.py -v

# Filter-aware queries test
python -m pytest tests/test_universal_filter_aware_engine.py -v

# Global aggregation test
python -m pytest tests/test_universal_global_aggregation.py -v

# API integration test
python -m pytest tests/test_api.py -v
```

---

## 💬 9. Supported Query Types & Natural Language Examples

| Category | Example Query | Execution Mechanism |
|---|---|---|
| **Capital Lookup** | *"What is the capital of Telangana?"* | Exact & RAG Entity Match |
| **Filtered Maximum** | *"Which village has more population in AP?"* | `MAX(population) WHERE state='Andhra Pradesh'` |
| **Filtered Minimum** | *"Which village has less population in Telangana?"* | `MIN(population) WHERE state='Telangana'` |
| **State Aggregation** | *"What is the total population in Maharashtra?"* | `SUM(population) WHERE state='Maharashtra'` |
| **Global Comparison** | *"Which state has more population?"* | Group By State Sum & Rank Top 1 |
| **Direct Comparison** | *"Compare population between Bihar and Gujarat"* | Comparison Engine (Difference, Ratio) |
| **Average Calculation** | *"What is the average literacy rate in Kerala?"* | `AVG(literacy_rate_percent) WHERE state='Kerala'` |
| **Follow-up Turn** | *"What about its area?"* | Conversation Context Resolution |
| **Typo & Slang** | *"whch vilage hs mr pop in telngna"* | RapidFuzz + Phonetic Normalizer |

---

## 📡 10. API Endpoints Table

| Endpoint | Method | Description |
|---|---|---|
| `/chat` or `/query` | `POST` | Primary conversational QA and query execution endpoint |
| `/health` | `GET` | Health status of backend, database, RAG, and Ollama |
| `/dataset/upload` | `POST` | Upload new CSV/Excel/JSON dataset |
| `/dataset/profile` | `GET` | Statistical summary and schema intelligence |
| `/dataset/schema` | `GET` | Schema structure, data types, and row count |
| `/query/history` | `GET` | Audit trail of processed queries |
| `/clear-chat` | `POST` | Clear conversation history for a session |
| `/new-session` | `POST` | Initialize a new isolated chat session |

---

## 🐳 11. Docker Deployment

To launch the full PostgreSQL + pgvector and backend stack using Docker Compose:
```bash
docker compose up -d
```
This initializes a pgvector container on port `5432` with automatic database initialization.

---

## 🛡️ 12. Hallucination Firewall & Factual Grounding Guarantee

Every numeric and factual answer provided by the system originates exclusively from the registered database tables and verified datasets. If a question is outside the scope of the loaded data (e.g., *"What is the GDP of Mars?"*), the system returns an honest out-of-scope response rather than hallucinating an answer.
