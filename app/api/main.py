"""
FastAPI application — main entry point.

Serves the HTML frontend at / and exposes the analysis API.
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_config

logger = logging.getLogger(__name__)
config = get_config()

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Financial Analyst API starting up…")
    try:
        from app.utils.llm import get_active_provider, get_active_model
        provider = get_active_provider()
        model    = get_active_model()
        logger.info("LLM provider: %s  model: %s", provider, model)
    except Exception as e:
        logger.warning("Could not check LLM status: %s", e)
    yield


app = FastAPI(
    title="Financial Analyst API",
    description=(
        "Multi-Agent Financial Analysis. Upload earnings PDFs to extract KPIs, "
        "analyse risks, generate summaries, and compare quarters."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS — allow everything (needed for local dev + HF Spaces)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
from app.api.routes import router  # noqa: E402
app.include_router(router)

# Serve frontend HTML at root
@app.get("/", include_in_schema=False)
async def serve_index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))

# Serve any other static assets from frontend/ (CSS, JS, images if added later)
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
