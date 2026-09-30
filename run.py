"""Application launch script with clear diagnostic checks for Dynamic AI Dataset Platform."""

import sys
import uvicorn
from app.config import settings
from app.dataset.loader import dataset_loader
from app.ai.ollama_client import ollama_client
from app.database.database import init_db
from app.utils.logger import logger


def run():
    print("=" * 70)
    print("   AI-Powered Dynamic Dataset Query & Retrieval Platform")
    print("=" * 70)

    # Initialize database
    init_db()

    # Active dataset
    df = dataset_loader.dataframe
    print(f"[*] Active Dataset : {dataset_loader.dataset_name} ({len(df)} rows, {len(df.columns)} columns)")
    print(f"[*] Columns        : {', '.join(dataset_loader.get_columns()[:6])}...")

    # Check Ollama
    o_info = ollama_client.check_health()
    if o_info.get("available"):
        print(f"[*] Ollama Status  : CONNECTED ({settings.OLLAMA_HOST})")
        print(f"[*] Configured LLM : {settings.OLLAMA_MODEL} (Model Installed: {o_info.get('model_installed')})")
    else:
        print(f"[*] Ollama Status  : OFFLINE / FALLBACK ACTIVE (Deterministic Engine Ready)")

    print(f"[*] Web Interface  : http://{settings.API_HOST}:{settings.API_PORT}")
    print(f"[*] API Docs       : http://{settings.API_HOST}:{settings.API_PORT}/docs")
    print("=" * 70)

    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
    )


if __name__ == "__main__":
    run()
