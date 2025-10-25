import logging
import time
import uuid
from collections.abc import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Logging middleware for reqeust/response
    Automatically logs all HTTP requests with timing and context
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate unique request id for tracing
        request_id = str(uuid.uuid4())

        # Add request_id to request state (accessible in route handlers)
        request.state.request_id = request_id

        #start timing
        start_time = time.time()

        #log incoming request
        logger.info(
            "Incoming request",
            extra={
                "extra_fields": {
                    "request_id": request_id,
                    "method": request.method,
                    "url": str(request.url),
                    "client_host": request.client.host if request.client else None,
                    "user_agent": request.headers.get("user-agent")
                }
            }
        )

        try:
            # Process requst
            response = await call_next(request)

            # Calculate duration
            duration = time.time() - start_time

            # Log response
            logger.info(
                "Request completed",
                extra={
                    "extra_fields":{
                        "request_id": request_id,
                        "method": request.method,
                        "url": str(request.url),
                        "status_code": response.status_code,
                        "duration_seconds": round(duration, 4)
                    }
                },
                exc_info=True # Inclde full stactrace
            )

            # Add request id to response headers (useful for debugging)
            response.headers["X-Request-ID"] = request_id

            return response

        except Exception as e:
            duration = time.time() - start_time

            logger.error(
                "Request Failed",
                extra = {
                    "extra_fields": {
                        "request_id": request_id,
                        "method": request.method,
                        "url": str(request.url),
                        "duration_seconds": round(duration, 4),
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
                },
                exc_info=True
            )
            raise
