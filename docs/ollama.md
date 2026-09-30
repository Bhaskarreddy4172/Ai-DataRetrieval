# Ollama Configuration Guide

## Supported Models
- `llama3.1:8b` (Default recommended)
- `mistral:7b`
- `qwen2.5:7b`
- `gemma3:7b`

## Setup Ollama Locally
1. Install Ollama from [ollama.com](https://ollama.com).
2. Pull preferred model:
   ```bash
   ollama pull llama3.1:8b
   ```
3. Set environment variable or configure `.env`:
   ```env
   OLLAMA_HOST=http://127.0.0.1:11434
   OLLAMA_MODEL=llama3.1:8b
   ```
4. Custom Modelfiles:
   Custom modelfiles with strict anti-hallucination system instructions are located in `ollama_models/`:
   ```bash
   ollama create dataset-llama3 -f ollama_models/Modelfile.llama3
   ```

