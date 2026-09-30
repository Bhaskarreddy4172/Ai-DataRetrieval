# Model Configuration & Modelfile Guide

## Overview

The Universal AI Dataset System utilizes **Ollama** as the local language understanding and query planning layer. The model serves solely as a tool selector and intent parser; it does NOT compute dataset metrics or guess facts.

---

## Ollama Provider Settings (`app/config.py` & `.env`)

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
OLLAMA_TIMEOUT=60
OLLAMA_MAX_RETRIES=2
OLLAMA_TEMPERATURE=0
```

---

## Recommended Ollama Models

| Model | Purpose | Recommendation |
|---|---|---|
| `llama3.1:8b` | Default general-purpose intent parser and tool selector | Highly Recommended |
| `qwen2.5:7b` / `qwen3:8b` | Excellent structured JSON output generation | Recommended |
| `mistral:7b` | Lightweight fast execution | Alternative |

---

## Startup Health Verification

At startup, the backend automatically performs health checks against Ollama `/api/tags`:
- `READY`: Ollama server online and configured model installed.
- `MODEL_NOT_FOUND`: Ollama server online but requested model missing.
- `UNAVAILABLE`: Ollama server offline or unreachable.
- `TIMEOUT` / `ERROR`: Network timeout or unexpected error.

If Ollama is unavailable or returns invalid output, the system gracefully falls back to deterministic NLP planning without crashing.

