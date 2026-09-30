"""Ollama integration service supporting llama3.1, mistral, qwen2.5, and gemma3."""

from app.ai.ollama_client import ollama_client, OllamaClient

__all__ = ["ollama_client", "OllamaClient"]

