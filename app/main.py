"""FastAPI server entry point, CORS middleware, and static asset mounts."""

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as api_router
from app.config import settings
from app.database.database import init_db
from app.utils.logger import logger

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(
    title=settings.APP_NAME,
    description="Dynamic AI Dataset Question-Answering and Retrieval Platform Using Ollama",
    version="2.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Router at root and /api
app.include_router(api_router, tags=["API"])
app.include_router(api_router, prefix="/api", tags=["API Prefix"])

# Static frontend files
dist_dir = FRONTEND_DIR / "dist"
assets_dir = dist_dir / "assets"
if assets_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

if FRONTEND_DIR.exists():
    css_dir = FRONTEND_DIR / "css"
    js_dir = FRONTEND_DIR / "js"
    if css_dir.exists():
        app.mount("/css", StaticFiles(directory=str(css_dir)), name="css")
    if js_dir.exists():
        app.mount("/js", StaticFiles(directory=str(js_dir)), name="js")

    @app.get("/", include_in_schema=False)
    def serve_frontend():
        if dist_dir.exists() and (dist_dir / "index.html").exists():
            return FileResponse(dist_dir / "index.html")
        return FileResponse(FRONTEND_DIR / "index.html")


@app.on_event("startup")
def on_startup():
    logger.info("Initializing application and database...")
    init_db()
    # Initialize ParentChildRegistry and run startup diagnostics
    try:
        from app.dataset.registry import parent_child_registry
        report = parent_child_registry.get_health_report()
        logger.info(
            f"STARTUP HEALTH CHECK -> MAIN DATASETS: {report['main_datasets']} | "
            f"PARENT ENTITIES: {report['parent_entities']} | "
            f"CHILD DATASETS DISCOVERED: {report['child_datasets_discovered']} | "
            f"CHILD DATASETS REGISTERED: {report['child_datasets_registered']} | "
            f"CHILD DATASETS FAILED: {report['child_datasets_failed']}"
        )
    except Exception as ex:
        logger.warning(f"ParentChildRegistry startup check warning: {ex}")

    # Auto-load default dataset if available
    try:
        from app.dataset.loader import dataset_loader
        default_file = Path(settings.DEFAULT_DATASET)
        if not default_file.exists():
            default_file = settings.UPLOADS_DIR / "indian_states_capitals.csv"
        if default_file.exists() and dataset_loader.dataframe is None:
            dataset_loader.load_dataset(default_file, custom_name="Indian States & Capitals")
    except Exception as ex:
        logger.warning(f"Default dataset auto-load warning: {ex}")

    # Pre-warm multi-child combined data and DuckDB in-memory table
    try:
        from app.dataset.multi_child_executor import multi_child_executor
        multi_child_executor.get_combined_dataframe(force_reload=True)
        from app.query.global_aggregation_engine import universal_global_aggregation_engine
        universal_global_aggregation_engine._ensure_duckdb()
        logger.info("Universal Global Aggregation Engine & DuckDB in-memory table pre-warmed successfully.")
    except Exception as ex:
        logger.warning(f"DuckDB table startup pre-warming warning: {ex}")

    logger.info("Application ready.")
