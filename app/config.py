"""Application configuration settings loaded from environment variables."""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class Settings:
    APP_NAME: str = "Dynamic AI Dataset Query & Retrieval Platform"
    API_HOST: str = os.getenv("API_HOST", "127.0.0.1")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "True").lower() in ("true", "1", "yes")

    # Centralized Ollama Configuration
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", OLLAMA_HOST)
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    OLLAMA_TIMEOUT: float = float(os.getenv("OLLAMA_TIMEOUT", "60.0"))
    OLLAMA_MAX_RETRIES: int = int(os.getenv("OLLAMA_MAX_RETRIES", "2"))
    OLLAMA_TEMPERATURE: float = float(os.getenv("OLLAMA_TEMPERATURE", "0.0"))

    PRIMARY_MODEL: str = os.getenv("PRIMARY_MODEL", OLLAMA_MODEL)
    SECONDARY_MODEL: Optional[str] = os.getenv("SECONDARY_MODEL", None)

    TOOL_TIMEOUT: float = float(os.getenv("TOOL_TIMEOUT", "30.0"))
    AGENT_TIMEOUT: float = float(os.getenv("AGENT_TIMEOUT", "120.0"))

    UPLOADS_DIR: Path = BASE_DIR / "uploads"
    DATA_DIR: Path = BASE_DIR / "data"
    DEFAULT_DATASET: str = os.getenv("DEFAULT_DATASET", str(DATA_DIR / "indian_states_capitals.csv"))
    DATABASE_PATH: str = os.getenv("DATABASE_PATH", str(BASE_DIR / "app.db"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")


settings = Settings()
settings.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
