"""
FastAPI application — main entry point.

Sets up the app, CORS, and includes all route handlers.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_config

logger = logging.getLogger(__name__)

config = get_config()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: runs startup logic, then yields for requests."""
    # --- Startup ---
    logger.info("Financial Analyst API starting up...")
    try:
        from app.utils.llm import check_ollama_running, get_active_model

        if check_ollama_running():
            model = get_active_model()
            logger.info("Ollama connected — active model: %s", model)
        else:
            logger.warning(
                "Ollama is not running. Start it with 'ollama serve' "
                "before submitting analysis requests."
            )
    except Exception as e:
        logger.warning("Could not check Ollama status: %s", e)

    yield  # Application runs here
    # --- Shutdown (nothing to clean up) ---


app = FastAPI(
    title="Financial Analyst API",
    description=(
        "Multi-Agent Financial Analysis platform. Upload earnings report PDFs "
        "to extract KPIs, analyse risks, generate summaries, and compare quarters."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# --- CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Include routes ---
from app.api.routes import router  # noqa: E402

app.include_router(router)
