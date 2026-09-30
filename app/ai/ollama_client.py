"""HTTP client for communicating with local Ollama instance with health caching."""

import time
from typing import Any, Dict, Optional
import requests
from app.config import settings
from app.utils.logger import logger


class OllamaClient:
    def __init__(
        self,
        host: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ):
        raw_host = (host or getattr(settings, "OLLAMA_BASE_URL", settings.OLLAMA_HOST)).rstrip("/")
        # Normalize localhost to 127.0.0.1 to avoid Windows IPv6 resolution latency
        self.host = raw_host.replace("localhost", "127.0.0.1")
        self.model = model or settings.OLLAMA_MODEL
        self.timeout = timeout or settings.OLLAMA_TIMEOUT
        self.max_retries = max_retries if max_retries is not None else settings.OLLAMA_MAX_RETRIES
        self._health_cache: Optional[Dict[str, Any]] = None
        self._health_cache_time: float = 0.0
        self._cache_ttl: float = 5.0  # seconds

    def check_health(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Check if Ollama server is responsive and verify model availability (cached)."""
        now = time.time()
        if not force_refresh and self._health_cache and (now - self._health_cache_time < self._cache_ttl):
            return self._health_cache

        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=1.0)
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("name", "") for m in data.get("models", [])]
                model_exists = any(self.model in m or m in self.model for m in models)
                status = "READY" if model_exists else "MODEL_NOT_FOUND"
                result = {
                    "status": status,
                    "available": model_exists,
                    "service_running": True,
                    "model_installed": model_exists,
                    "models": models,
                    "selected_model": self.model,
                }
            else:
                result = {
                    "status": "UNAVAILABLE",
                    "available": False,
                    "service_running": False,
                    "model_installed": False,
                    "error": f"HTTP {resp.status_code}",
                }
        except requests.exceptions.Timeout:
            result = {
                "status": "TIMEOUT",
                "available": False,
                "service_running": False,
                "model_installed": False,
                "error": "Request timed out",
            }
        except requests.exceptions.ConnectionError:
            result = {
                "status": "UNAVAILABLE",
                "available": False,
                "service_running": False,
                "model_installed": False,
                "error": "Connection refused",
            }
        except Exception as e:
            result = {
                "status": "ERROR",
                "available": False,
                "service_running": False,
                "model_installed": False,
                "error": str(e),
            }

        self._health_cache = result
        self._health_cache_time = now
        return result

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        json_format: bool = False,
        temperature: float = 0.0,
        num_ctx: int = 4096,
        retries: Optional[int] = None,
    ) -> Optional[str]:
        """Call Ollama /api/generate endpoint synchronously with retry policy."""
        max_attempts = (retries if retries is not None else self.max_retries) + 1
        url = f"{self.host}/api/generate"
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": num_ctx,
            },
        }
        if system:
            payload["system"] = system
        if json_format:
            payload["format"] = "json"

        for attempt in range(max_attempts):
            try:
                resp = requests.post(url, json=payload, timeout=self.timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get("response", "").strip()
                else:
                    logger.warning(
                        f"Ollama returned HTTP {resp.status_code} (attempt {attempt + 1}/{max_attempts}): {resp.text}"
                    )
            except requests.exceptions.Timeout:
                logger.warning(
                    f"Ollama request timed out after {self.timeout}s (attempt {attempt + 1}/{max_attempts})."
                )
            except requests.exceptions.ConnectionError:
                logger.warning(
                    f"Could not connect to Ollama at {self.host} (attempt {attempt + 1}/{max_attempts})."
                )
            except Exception as e:
                logger.error(
                    f"Unexpected error communicating with Ollama (attempt {attempt + 1}/{max_attempts}): {e}"
                )

            if attempt < max_attempts - 1:
                time.sleep(0.1 * (attempt + 1))

        return None

    def generate_stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.0,
        num_ctx: int = 4096,
    ):
        """Yield streaming response tokens from Ollama /api/generate."""
        import json
        url = f"{self.host}/api/generate"
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_ctx": num_ctx,
            },
        }
        if system:
            payload["system"] = system

        try:
            with requests.post(url, json=payload, stream=True, timeout=self.timeout) as resp:
                if resp.status_code == 200:
                    for line in resp.iter_lines():
                        if line:
                            chunk = json.loads(line.decode("utf-8"))
                            token = chunk.get("response", "")
                            if token:
                                yield token
                            if chunk.get("done", False):
                                break
        except Exception as e:
            logger.warning(f"Ollama streaming failed: {e}")

    def set_model(self, model_name: str) -> None:
        """Dynamically switch active Ollama model (llama3.1, mistral, qwen2.5, gemma3)."""
        self.model = model_name
        self._health_cache = None
        logger.info(f"Ollama active model switched to '{model_name}'.")


ollama_client = OllamaClient()
