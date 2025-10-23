from decimal import Context
import logging
from string import Formatter
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import json
from datetime import datetime,timezone
from typing import Any, Dict


LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

class JsonFormatter(logging.Formatter):
    """
    Structured logging in JSON format
    This makes logs machine readable for tools like CloudWatch
    """

    def format(self, record: logging.LogRecord) -> str:

        log_data: Dict[str, Any] = {
            "timestamp" : datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        #add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        #add extra fields if available
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields) #not sure if record has this filed


        return json.dumps(log_data, default=str )   



class ContextFilter(logging.Filter):
    """
    Add contextual information to every log.
    Request id for tracing requests across services.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = getattr(record, "request_id", "no-request-id")
        return True


def setup_logging(
    log_level: str = "INFO",
    json_logs: bool = True,
    log_file: bool = True
) -> None:

    """
    Setup application logging

    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_logs: use JSON formatting
        log_file: write logs to file
    """

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))


    #remove existing handlers
    root_logger.handlers.clear()

    if json_logs:
        formatter=JsonFormatter()
    else:
        formatter = logging.Formatter(
            fmt = "%(asctime)s - %(name)s - %(levelname)s -%(message)s -[%(filename)s:%(lineno)d]",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

    #Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(ContextFilter())
    root_logger.addHandler(console_handler)


    if log_file:
        # application logs - Rotating by size
        app_handler = RotatingFileHandler(
            LOGS_DIR / "app.log",
            maxBytes=10*1024*1024, #10MB
            backupCount=5, #keep 5 backup files
            encoding="utf-8" 
        )
        app_handler.setLevel(logging.DEBUG)
        app_handler.setFormatter(formatter)
        app_handler.addFilter(ContextFilter())
        root_logger.addHandler(app_handler)


        #Error logs - Rotating by time
        error_handler = TimedRotatingFileHandler(
            LOGS_DIR / "error.log",
            when="midnight", 
            interval=1,
            backupCount=30,
            encoding="utf-8"
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        error_handler.addFilter(ContextFilter())
        root_logger.addHandler(error_handler)

    #silence noisy third party logs
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)



def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.
    Use __name__ as the parameter
    """

    return logging.getLogger(name)

class LoggerAdapter(logging.LoggerAdapter):
    """
    Adapter to add extra context to all logs from a logger.
    """
    def process(self, msg: str, kwargs: Any) -> tuple:
        #merge extra fields
        if "extra" not in kwargs:
            kwargs["extra"] = {}

        if "extra_fields" not in kwargs["extra"]:
            kwargs["extra"]["extra_fields"] = {}

        kwargs["extra"]["extra_fields"].update(self.extra)

        return msg, kwargs