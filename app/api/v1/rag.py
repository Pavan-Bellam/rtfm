from turtle import ht
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import ValidationError

from app.core.logging import get_logger
from app.core.db import get_db
from app.schema.rag import ScrapeRequest, ScrapeResponse, IngestionRequest, IngestionResponse
from app.services.rag.scrape import scrape as scrape_service
from app.services.rag.ingest import ingest as ingest_service

router = APIRouter()

logger = get_logger(__name__)


@router.post(
    "/scrape",
    description="Crawl web from starting from the url sent in the request and scrape `limit` number of pages",
    response_model=ScrapeResponse,
    status_code=status.HTTP_200_OK
)
async def scrape(req: ScrapeRequest):
    """
    API endpoint to scrape web pages and convert them to markdown.
    Crawls starting from the provided URL and saves results to organized directories.

    Args:
        req: ScrapeRequest containing tool_name, url, and page limit

    Returns:
        ScrapeResponse: Paths to saved markdown and metadata directories

    Raises:
        HTTPException 400: For validation errors or invalid input parameters
        HTTPException 500: For server errors during scraping or file operations
        HTTPException 503: For external service (Firecrawl) unavailability
    """
    logger.info("Received scrape request", extra={
        "extra_fields": {
            "tool_name": req.tool_name,
            "url": str(req.url),
            "limit": req.limit
        }
    })

    try:
        # Call the scrape service
        result = await scrape_service(
            tool_name=req.tool_name,
            url=str(req.url),
            limit=req.limit
        )

        logger.info("Scrape request completed successfully", extra={
            "extra_fields": {
                "tool_name": req.tool_name,
                "url": str(req.url),
                "markdown_dir": str(result["markdown_dir"]),
                "metadata_dir": str(result["metadata_dir"])
            }
        })

        return ScrapeResponse(
            markdown_dir=str(result["markdown_dir"]),
            metadata_dir=str(result["metadata_dir"])
        )

    except ValidationError as e:
        logger.error("Validation error in scrape request", exc_info=True, extra={
            "extra_fields": {
                "tool_name": req.tool_name,
                "url": str(req.url),
                "error_type": "ValidationError",
                "validation_errors": str(e.errors())
            }
        })
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Validation error: {str(e)}"
        ) from e

    except ValueError as e:
        logger.error("Invalid parameter in scrape request", exc_info=True, extra={
            "extra_fields": {
                "tool_name": req.tool_name,
                "url": str(req.url),
                "error_type": "ValueError"
            }
        })
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid request parameters: {str(e)}"
        ) from e

    except OSError as e:
        logger.error("File system error during scrape operation", exc_info=True, extra={
            "extra_fields": {
                "tool_name": req.tool_name,
                "url": str(req.url),
                "error_type": "OSError"
            }
        })
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save scraped data due to file system error"
        ) from e

    except ConnectionError as e:
        logger.error("Connection error to Firecrawl service", exc_info=True, extra={
            "extra_fields": {
                "tool_name": req.tool_name,
                "url": str(req.url),
                "error_type": "ConnectionError"
            }
        })
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to connect to scraping service. Please try again later."
        ) from e

    except TimeoutError as e:
        logger.error("Timeout during scrape operation", exc_info=True, extra={
            "extra_fields": {
                "tool_name": req.tool_name,
                "url": str(req.url),
                "error_type": "TimeoutError"
            }
        })
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Scraping operation timed out. Please try with a smaller limit."
        ) from e

    except Exception as e:
        logger.error("Unexpected error processing scrape request", exc_info=True, extra={
            "extra_fields": {
                "tool_name": req.tool_name,
                "url": str(req.url),
                "limit": req.limit,
                "error_type": type(e).__name__
            }
        })
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing your request"
        ) from e


@router.post(
    '/ingest',
    description="Crawls from the base url, save webpages as files, chunks them and saves metadata and embedding in db",
    response_model=IngestionResponse,
    status_code=status.HTTP_200_OK
)
async def ingest_data(req: IngestionRequest, db: AsyncSession = Depends(get_db)):
    """
    API endpoint to ingest web content: scrape, chunk, embed, and store in database.

    Args:
        req: IngestionRequest containing tool_name, url, and page limit
        db: Database session

    Returns:
        IngestionResponse: Statistics about the ingestion process

    Raises:
        HTTPException 400: For validation errors or invalid input parameters
        HTTPException 500: For server errors during ingestion
    """
    logger.info("Received ingest request", extra={
        "extra_fields": {
            "tool_name": req.tool_name,
            "url": str(req.url),
            "limit": req.limit
        }
    })

    try:
        result = await ingest_service(
            tool_name=req.tool_name,
            url=str(req.url),
            limit=req.limit,
            db=db
        )

        logger.info("Ingest request completed successfully", extra={
            "extra_fields": {
                "tool_name": req.tool_name,
                "url": str(req.url),
                "limit": req.limit,
                **result
            }
        })

        # Determine success message
        if result['files_complete_failures'] == 0 and result['files_partial_failures'] == 0:
            message = "Ingestion completed successfully"
        elif result['files_complete_failures'] > 0 and result['files_success'] == 0:
            message = "Ingestion completed with failures"
        else:
            message = "Ingestion completed with partial failures"

        return IngestionResponse(
            message=message,
            pages_scraped=result['pages_scraped'],
            files_processed=result['files_processed'],
            files_success=result['files_success'],
            files_partial_failures=result['files_partial_failures'],
            files_complete_failures=result['files_complete_failures'],
            total_chunks_created=result['total_chunks_created'],
            markdown_dir=result['markdown_dir'],
            metadata_dir=result['metadata_dir']
        )

    except ValueError as ve:
        logger.error("Invalid parameter in ingest request", exc_info=True, extra={
            "extra_fields": {
                "tool_name": req.tool_name,
                "url": str(req.url),
                "error_type": "ValueError"
            }
        })
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )

    except Exception as e:
        logger.error("Unexpected error processing ingest request", exc_info=True, extra={
            "extra_fields": {
                "tool_name": req.tool_name,
                "url": str(req.url),
                "limit": req.limit,
                "error_type": type(e).__name__
            }
        })
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during ingestion"
        )
