"""FastAPI application factory and entry point."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as api_router
from app.config import settings
from app.logging_config import setup_logging

logger = logging.getLogger("sledge.extractor")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manages application startup and shutdown lifecycle events."""
    setup_logging(level=settings.log_level)
    logger.info(
        "Starting Construction Change-Order Extraction Service (model=%s, max_pages=%d)",
        settings.openai_model,
        settings.max_pdf_pages,
    )
    yield
    logger.info("Shutting down Construction Change-Order Extraction Service.")


def create_app() -> FastAPI:
    """Builds and configures the FastAPI application."""
    application = FastAPI(
        title="Construction Change-Order Extraction API",
        description=(
            "Reliable extraction pipeline for extracting structured data, "
            "verifiable evidence, and confidence scores from construction change orders in PDF or raw text."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(api_router)

    return application


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
