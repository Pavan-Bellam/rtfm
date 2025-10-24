import json
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging

from app.core.logging import setup_logging, get_logger
from app.middleware.logging import LoggingMiddleware
from app.core.settings import settings
from app.api.v1.rag import router as rag_router
from app.api.v1.health import router as health_router

# Set up logging
#make these values dependent on env(Todo)

setup_logging(
    log_level=settings.LOG_LEVEL, 
    json_logs=settings.JSON_LOGS, 
    log_file=settings.LOG_FILE
)

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifesapn events for startup and shutdown. Will be configured later if required(ToDo)
    """
    # Start up
    logger.info("Application starting up")
    
    yield

    # Shutdown
    logger.info("Application shutting down")


app = FastAPI(
    title = "rtfm.ai",
    version = "0.1.0",
    lifespan=lifespan
)

#Add logging middleware
app.add_middleware(LoggingMiddleware)

# Include routers
app.include_router(rag_router, prefix="/api/v1/rag", tags=["RAG"])
app.include_router(health_router, prefix="/api/v1/health", tags=["Health"])

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catch all unhandled exceptions and log them.
    """
    logger.error(
        "Unhandled exception",
        extra={
            "extra_fields": {
                "request_id": getattr(request.state, "request_id", "unknown"),
                "path": request.url.path,
                "method": request.method,
                "error_type": type(exc).__name__,
            }
        },
        exc_info=True
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "request_id": getattr(request.state, "request_id", "unknown")
        }
    )
