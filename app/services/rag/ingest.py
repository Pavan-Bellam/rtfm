from app.services.rag.scrape import scrape
from app.services.rag.chunk import chunk_files
from app.services.rag.utils import number_files
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.logging import get_logger


logger = get_logger(__name__)

async def ingest(
    tool_name: str,
    url: str,
    limit: int,
    db: AsyncSession,
):
    """
    Ingest data from a URL by scraping web pages and chunking them into the database.
    Orchestrates the scraping and chunking pipeline.

    Args:
        tool_name: Name identifier for organizing scraped data
        url: Starting URL for web crawling
        limit: Maximum number of pages to scrape
        db: AsyncSession to the database

    Returns:
        dict: Contains scraping and chunking statistics including pages_scraped,
              files_processed, success/failure counts, total_chunks_created, and directory paths

    Raises:
        ValueError: If tool_name is empty, limit <= 0, paths are invalid, or file counts mismatch
        FileNotFoundError: If markdown/metadata files or directories don't exist
        PermissionError: If directory access is denied
        OSError: If file system operations fail
        UnicodeEncodeError: If documents contain invalid unicode during saving
        UnicodeDecodeError: If file reading encounters encoding errors
        json.JSONDecodeError: If metadata JSON is malformed
        Exception: For Firecrawl API errors, network issues, chunking, or database errors
    """
    logger.info("Starting ingestion pipeline", extra={
        "extra_fields": {
            "tool_name": tool_name,
            "url": url,
            "limit": limit
        }
    })

    try:
        # Scrape web pages
        logger.info("Starting scraping phase", extra={
            "extra_fields": {"tool_name": tool_name, "url": url, "limit": limit}
        })
        scrape_result = await scrape(tool_name, url, limit)
        markdown_files_dir = scrape_result['markdown_dir']
        metadata_files_dir = scrape_result['metadata_dir']
        pages_scraped = scrape_result['pages_scraped']
        logger.info("Scraping phase completed", extra={
            "extra_fields": {"tool_name": tool_name, "pages_scraped": pages_scraped}
        })

        # Number files
        await number_files(files_dir=markdown_files_dir)

        # Chunk and embed files
        logger.info("Starting chunking and embedding phase", extra={
            "extra_fields": {"tool_name": tool_name}
        })
        chunk_result = await chunk_files(db, metadata_files_dir=metadata_files_dir, markdown_files_dir=markdown_files_dir)

        logger.info("Chunking and embedding phase completed", extra={
            "extra_fields": {"tool_name": tool_name, **chunk_result}
        })

        logger.info("Ingestion pipeline completed successfully", extra={
            "extra_fields": {
                "tool_name": tool_name,
                "url": url,
                "limit": limit,
                "pages_scraped": pages_scraped,
                **chunk_result
            }
        })

        # Return aggregated results
        return {
            "pages_scraped": pages_scraped,
            "files_processed": chunk_result['success'] + chunk_result['partial_failures'] + chunk_result['complete_failures'],
            "files_success": chunk_result['success'],
            "files_partial_failures": chunk_result['partial_failures'],
            "files_complete_failures": chunk_result['complete_failures'],
            "total_chunks_created": chunk_result['total_chunks_created'],
            "markdown_dir": str(markdown_files_dir),
            "metadata_dir": str(metadata_files_dir)
        }

    except Exception as e:
        logger.error("Ingestion pipeline failed", exc_info=True, extra={
            "extra_fields": {
                "tool_name": tool_name,
                "url": url,
                "limit": limit,
                "error_type": type(e).__name__
            }
        })
        raise