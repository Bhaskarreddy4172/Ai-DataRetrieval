"""Model Manager singleton for active LLM provider registration and dynamic switching."""

from typing import Dict, Optional
from app.llm.base import LLMProvider
from app.llm.ollama_provider import OllamaProvider
from app.utils.logger import logger


class ModelManager:
    """Registry managing active LLM providers and dynamic model switching."""

    def __init__(self):
        self._providers: Dict[str, LLMProvider] = {}
        self._default_provider: Optional[LLMProvider] = None
        self._active_model_name: str = "llama3.1:8b"

        # Register default Ollama provider
        ollama_prov = OllamaProvider(self._active_model_name)
        self.register_provider("ollama", ollama_prov)
        self._default_provider = ollama_prov

    def register_provider(self, name: str, provider: LLMProvider) -> None:
        self._providers[name.lower()] = provider

    def get_provider(self, name: Optional[str] = None) -> LLMProvider:
        if name and name.lower() in self._providers:
            return self._providers[name.lower()]
        return self._default_provider or OllamaProvider()

    def set_active_model(self, model_name: str) -> None:
        self._active_model_name = model_name
        prov = self.get_provider("ollama")
        if isinstance(prov, OllamaProvider):
            prov = OllamaProvider(model_name)
            self.register_provider("ollama", prov)
            self._default_provider = prov
        logger.info(f"ModelManager set active model to '{model_name}'.")

    @property
    def active_model(self) -> str:
        return self._active_model_name


model_manager = ModelManager()

