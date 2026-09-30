# Ollama Models Setup Guide

The Universal Dataset Chatbot supports flexible local LLM backends via Ollama:
- `llama3.1:8b` (Default recommended)
- `mistral:7b`
- `qwen2.5:7b`
- `gemma3:7b`

## Responsibility Division
- **OLLAMA**: Handles natural language query normalization, conversational explanations of verified results, and semantic understanding.
- **PANDAS / DUCKDB**: Deterministically performs all filtering, grouping, counting, aggregations, and calculations. Ollama is **never** permitted to calculate numbers or fabricate facts.

## Commands
```bash
# Pull base models
ollama pull llama3.1:8b
ollama pull mistral:7b
ollama pull qwen2.5:7b

# Create custom grounded personas
ollama create dataset-llama3 -f Modelfile.llama3
ollama create dataset-mistral -f Modelfile.mistral
ollama create dataset-qwen -f Modelfile.qwen2.5
```

