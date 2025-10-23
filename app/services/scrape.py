import asyncio
from firecrawl.v2.types import Document
from app.core import settings
from app.core.deps import firecrawl
from app.core.settings import settings
from app.core.logging import get_logger, setup_logging
import json
from pathlib import Path
import hashlib
import time

setup_logging(
    log_level=settings.LOG_LEVEL,
    json_logs=settings.JSON_LOGS,
    log_file=settings.LOG_FILE
)

logger = get_logger(__name__)


def save_document(idx: int, obj: Document, markdown_dir: Path, metadata_dir: Path) -> None:
    """
    Save a scraped document to markdown and JSON metadata files.
    Creates unique filenames based on document URL hash and sanitized title.

    Args:
        idx: Document index in the crawl results
        obj: Document object containing markdown content and metadata
        markdown_dir: Directory path for saving markdown files
        metadata_dir: Directory path for saving metadata JSON files

    Raises:
        OSError: If file writing fails due to permissions or disk issues
        UnicodeEncodeError: If document contains invalid unicode characters
    """
    try:
        url = obj.metadata.url or f"doc_{idx}"
        doc_id = hashlib.md5(url.encode()).hexdigest()[:12]

        safe_title = obj.metadata.title or f"doc_{idx}"
        safe_title = safe_title.replace("/", "-").replace(r"\\", "-").replace(":", "-")[:50]
        base_filename = f"{idx:04d}_{doc_id}_{safe_title}"

        logger.debug(f"Saving document {idx}: {base_filename}", extra={
            "extra_fields": {
                "doc_index": idx,
                "doc_id": doc_id,
                "url": url,
                "title": obj.metadata.title
            }
        })

        # Save markdown file
        markdown_file = markdown_dir / f"{base_filename}.md"
        with open(markdown_file, "w", encoding="utf-8") as f:
            f.write(obj.markdown or "")

        # Save metadata as JSON
        json_obj = {
            'markdown': obj.markdown,
            "metadata": {
                "title": obj.metadata.title,
                "description": obj.metadata.description,
                "url": obj.metadata.url,
                "keywords": obj.metadata.keywords
            }
        }

        metadata_file = metadata_dir / f"{base_filename}.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(json_obj, f, indent=2, ensure_ascii=False)

        logger.debug(f"Successfully saved document {idx}", extra={
            "extra_fields": {
                "markdown_file": str(markdown_file),
                "metadata_file": str(metadata_file)
            }
        })

    except UnicodeEncodeError as e:
        logger.error(f"Failed to save document {idx} due to unicode encoding error", exc_info=True, extra={
            "extra_fields": {
                "doc_index": idx,
                "url": url,
                "error_type": "UnicodeEncodeError",
                "encoding": e.encoding,
                "problematic_position": e.start
            }
        })
        raise
    except OSError as e:
        logger.error(f"Failed to save document {idx} due to file system error", exc_info=True, extra={
            "extra_fields": {
                "doc_index": idx,
                "url": url,
                "error_type": "OSError"
            }
        })
        raise
    except Exception as e:
        logger.error(f"Unexpected error saving document {idx}", exc_info=True, extra={
            "extra_fields": {
                "doc_index": idx,
                "error_type": type(e).__name__
            }
        })
        raise

async def scrape(tool_name: str, url: str, limit: int):
    """
    Crawl and scrape web pages starting from a given URL.
    Downloads pages, converts to markdown, and saves along with metadata.

    Args:
        tool_name: Name identifier for organizing scraped data in directories
        url: Starting URL for web crawling
        limit: Maximum number of pages to scrape

    Returns:
        dict: Contains 'markdown_dir' and 'metadata_dir' paths as Path objects

    Raises:
        ValueError: If parameters are invalid (empty tool_name, invalid URL, limit <= 0)
        OSError: If directory creation or file writing fails
        Exception: For Firecrawl API errors or network issues
    """
    logger.info(f"Starting web scraping operation", extra={
        "extra_fields": {
            "url": url,
            "limit": limit,
            "tool_name": tool_name
        }
    })

    # Validate inputs
    if not tool_name or not tool_name.strip():
        logger.error("Invalid tool_name: cannot be empty")
        raise ValueError("tool_name cannot be empty")

    if limit <= 0:
        logger.error(f"Invalid limit: {limit}, must be greater than 0")
        raise ValueError(f"limit must be greater than 0, got {limit}")

    try:
        # Crawl web pages
        start_time = time.time()
        logger.info(f"Starting Firecrawl crawl operation", extra={
            "extra_fields": {"url": url, "limit": limit}
        })

        docs = await firecrawl.crawl(
            url=url,
            limit=limit,
            scrape_options={'formats': ['markdown']}
        )

        scraping_duration = time.time() - start_time
        docs_count = len(docs.data) if docs and docs.data else 0

        logger.info(f"Crawling completed successfully", extra={
            "extra_fields": {
                "duration_seconds": round(scraping_duration, 4),
                "pages_crawled": docs_count,
                "url": url
            }
        })

        if docs_count == 0:
            logger.warning(f"No documents were crawled from URL", extra={
                "extra_fields": {"url": url, "limit": limit}
            })

        # Prepare storage directories
        logger.info(f"Preparing storage directories for scraped data")
        saving_start_time = time.time()
        data_dir = settings.RAW_DATA_STORAGE_URL
        markdown_dir = data_dir / tool_name / "markdown"
        metadata_dir = data_dir / tool_name / "metadata"

        try:
            markdown_dir.mkdir(parents=True, exist_ok=True)
            metadata_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Storage directories created", extra={
                "extra_fields": {
                    "markdown_dir": str(markdown_dir),
                    "metadata_dir": str(metadata_dir)
                }
            })
        except OSError as e:
            logger.error(f"Failed to create storage directories", exc_info=True, extra={
                "extra_fields": {
                    "markdown_dir": str(markdown_dir),
                    "metadata_dir": str(metadata_dir),
                    "error_type": "OSError"
                }
            })
            raise

        # Save documents concurrently
        logger.info(f"Starting parallel document save operation", extra={
            "extra_fields": {"document_count": docs_count}
        })

        save_tasks = [
            asyncio.to_thread(save_document, idx, obj, markdown_dir, metadata_dir)
            for idx, obj in enumerate(docs.data)
        ]

        results = await asyncio.gather(*save_tasks, return_exceptions=True)

        # Check for any failed saves
        failed_saves = [i for i, r in enumerate(results) if isinstance(r, Exception)]
        if failed_saves:
            logger.error(f"Failed to save {len(failed_saves)} documents", extra={
                "extra_fields": {
                    "failed_indices": failed_saves,
                    "total_documents": docs_count
                }
            })
            # Re-raise the first exception
            raise results[failed_saves[0]]

        saving_duration = time.time() - saving_start_time
        logger.info(f"Document saving completed successfully", extra={
            "extra_fields": {
                "total_documents": docs_count,
                "duration_seconds": round(saving_duration, 4),
                "markdown_dir": str(markdown_dir),
                "metadata_dir": str(metadata_dir)
            }
        })

        return {
            "markdown_dir": markdown_dir,
            "metadata_dir": metadata_dir
        }

    except ValueError as e:
        logger.error(f"Validation error in scrape operation", exc_info=True, extra={
            "extra_fields": {"error_type": "ValueError"}
        })
        raise
    except OSError as e:
        logger.error(f"File system error during scrape operation", exc_info=True, extra={
            "extra_fields": {"error_type": "OSError"}
        })
        raise
    except Exception as e:
        logger.error(f"Unexpected error during scrape operation", exc_info=True, extra={
            "extra_fields": {
                "url": url,
                "tool_name": tool_name,
                "limit": limit,
                "error_type": type(e).__name__
            }
        })
        raise


if __name__ == "__main__":

    print("starting asyncio run")
    asyncio.run(scrape("fastapi", url="https://fastapi.tiangolo.com/", limit=1))