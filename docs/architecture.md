# Architecture Documentation

## Universal Dataset Chatbot Architecture

The Universal Dataset Chatbot is engineered with a strict **Grounded Execution Pipeline** where customer dataset answers are never guessed or hallucinated by LLMs. 

### Core Flow Diagram

```text
                           CUSTOMER QUESTION
                                   │
                                   ▼
                       ┌───────────────────────┐
                       │ Language Normalizer   │ (Hinglish, Telugu-Eng, Butler English)
                       └───────────┬───────────┘
                                   ▼
                       ┌───────────────────────┐
                       │ Spell & Typo Engine   │ (RapidFuzz, Levenshtein, Metaphone)
                       └───────────┬───────────┘
                                   ▼
                       ┌───────────────────────┐
                       │ Entity & Alias Engine │ (Acronyms, Airport codes, Acronyms)
                       └───────────┬───────────┘
                                   ▼
                       ┌───────────────────────┐
                       │ Hybrid Vector Store   │ (Dense embeddings + Lexical RRF)
                       └───────────┬───────────┘
                                   ▼
                       ┌───────────────────────┐
                       │ Semantic Schema Mapper│ (Zero-shot column binding)
                       └───────────┬───────────┘
                                   ▼
                       ┌───────────────────────┐
                       │ Query Planner         │ (JSON Query DSL, Never Raw SQL)
                       └───────────┬───────────┘
                                   ▼
                       ┌───────────────────────┐
                       │ Query Plan Validator  │ (Schema, Type, Operator checks)
                       └───────────┬───────────┘
                                   ▼
                       ┌───────────────────────┐
                       │ Deterministic Engine  │ (Pandas / DuckDB execution)
                       └───────────┬───────────┘
                                   ▼
                       ┌───────────────────────┐
                       │ Hallucination Firewall│ (Fact-check against Dataframe)
                       └───────────┬───────────┘
                                   ▼
                       ┌───────────────────────┐
                       │ Ollama NL Formatter   │ (llama3.1 / mistral / qwen2.5)
                       └───────────┬───────────┘
                                   ▼
                            VERIFIED ANSWER
```

### Responsibility Separation
1. **OLLAMA LLM**:
   - Natural language comprehension.
   - Dialogue intent contextualization.
   - Conversational synthesis of verified records.
2. **PANDAS & DUCKDB**:
   - Exact mathematical calculations (SUM, AVG, MIN, MAX).
   - Strict filtering, grouping, sorting, and counting.
   - Zero hallucination guarantee.
3. **DATASET**:
   - The single factual source of truth.

