import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from agents.agents.chunk import ChunkingAgent
from app.core.logging import get_logger, setup_logging
from app.core.rag_config import openai_embeddings, chunk_config_model
from app.core.settings import settings
from app.models.chunks import Chunk
from app.services.db.chunk import create_chunk
from app.services.rag.utils import extract_line_range

setup_logging(
    log_level=settings.LOG_LEVEL,
    json_logs=settings.JSON_LOGS,
    log_file=settings.LOG_FILE
)

logger = get_logger(__name__)

chunkning_agent = ChunkingAgent(
    provider=chunk_config_model.get('provider', 'openai'),
    model=chunk_config_model.get('model', 'gpt-5-nano'),
)

async def chunk_file(markdown_file_path: Path, metadata_file_path: Path, db: AsyncSession) -> Any:
    """
    Process a markdown file and its metadata to create chunks in the database.
    Reads the file, generates semantic chunks using AI, and stores them with embeddings.

    Args:
        markdown_file_path: Path to the markdown file to chunk
        metadata_file_path: Path to the metadata JSON file
        db: AsyncSession to the database

    Returns:
        dict: Contains 'total_chunks' and 'failed_chunks' counts

    Raises:
        ValueError: If paths are not files or db is None
        FileNotFoundError: If markdown or metadata files don't exist
        UnicodeDecodeError: If file reading encounters encoding errors
        OSError: If file reading fails due to permissions or disk issues
        json.JSONDecodeError: If metadata JSON is malformed
        Exception: For AI chunking, embedding generation, or database errors
    """
    # Convert to Path objects if needed
    if not isinstance(markdown_file_path, Path):
        markdown_file_path = Path(markdown_file_path)

    if not isinstance(metadata_file_path, Path):
        metadata_file_path = Path(metadata_file_path)
    
    # Validate inputs
    if not markdown_file_path.exists():
        logger.error("Markdown file does not exist", extra={
            "extra_fields": {"file_path": str(markdown_file_path)}
        })
        raise FileNotFoundError(f"{markdown_file_path} does not exist")

    if not markdown_file_path.is_file():
        logger.error("Markdown path is not a file", extra={
            "extra_fields": {"file_path": str(markdown_file_path)}
        })
        raise ValueError(f"{markdown_file_path} is not a file")

    if not metadata_file_path.exists():
        logger.error("Metadata file does not exist", extra={
            "extra_fields": {"file_path": str(metadata_file_path)}
        })
        raise FileNotFoundError(f"{metadata_file_path} does not exist")

    if not metadata_file_path.is_file():
        logger.error("Metadata path is not a file", extra={
            "extra_fields": {"file_path": str(metadata_file_path)}
        })
        raise ValueError(f"{metadata_file_path} is not a file")

    if db is None:
        logger.error("Database session cannot be None")
        raise ValueError("db session cannot be None")


    # Read markdown file
    start_time = time.time()
    try:
        with open(markdown_file_path, 'r', encoding='utf-8') as f:
            data = f.read()
    except (UnicodeDecodeError, OSError) as e:
        logger.error("Failed to read markdown file", exc_info=True, extra={
            "extra_fields": {
                "file_path": str(markdown_file_path),
                "error_type": type(e).__name__
            }
        })
        raise

    # Read metadata file
    try:
        with open(metadata_file_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
        logger.error("Failed to read or parse metadata file", exc_info=True, extra={
            "extra_fields": {
                "file_path": str(metadata_file_path),
                "error_type": type(e).__name__
            }
        })
        raise

    # Generate chunks using AI
    try:
        result = await chunkning_agent.run(data)
    except Exception as e:
        logger.error("Failed to generate chunks with AI agent", exc_info=True, extra={
            "extra_fields": {
                "file_path": str(markdown_file_path),
                "error_type": type(e).__name__
            }
        })
        raise

    chunks_generated = len(result['chunks'])

    if chunks_generated == 0:
        logger.warning("No chunks generated", extra={
            "extra_fields": {
                "file_path": str(markdown_file_path)
            }
        })

    # Process and store chunks
    failed_chunks = 0
    embedding_start_time = time.time()

    for idx, chunk in enumerate(result['chunks']):
        try:
            chunk_content = extract_line_range(data, chunk.start_line_number, chunk.end_line_number)
            chunk_context = chunk.context
            cleaned_chunk_content = re.sub(r"^#\d+:\s*", "", chunk_content, flags=re.MULTILINE)
            final_text_to_embed = chunk_context + "\n\n" + cleaned_chunk_content

            embedding = openai_embeddings.embed_query(final_text_to_embed)

            chunk_id_in_db = await create_chunk(
                tool_name=metadata.get('tool_name', None),
                url=metadata.get('url', None),
                content=cleaned_chunk_content,
                context=chunk_context,
                embedding=embedding,
                db=db
            )

        except Exception as e:
            failed_chunks += 1
            logger.error(f"Failed to create chunk {idx + 1}", exc_info=True, extra={
                "extra_fields": {
                    "chunk_index": idx,
                    "file_path": str(markdown_file_path),
                    "error_type": type(e).__name__
                }
            })

    return {
        "total_chunks": chunks_generated,
        "failed_chunks": failed_chunks
    }


async def chunk_files(
    db: AsyncSession,
    markdown_files_list: Optional[list[Path]] = None,
    metadata_files_list: Optional[list[Path]] = None,
    metadata_files_dir: Optional[Path] = None,
    markdown_files_dir: Optional[Path] = None
) -> Any:
    """
    Process multiple markdown files and their metadata to create chunks in the database.
    Accepts either file lists or directory paths to discover files.

    Args:
        db: AsyncSession to the database
        markdown_files_list: Optional list of markdown file paths
        metadata_files_list: Optional list of metadata file paths
        metadata_files_dir: Optional directory containing metadata files
        markdown_files_dir: Optional directory containing markdown files

    Returns:
        dict: Contains 'success', 'partial_failures', 'complete_failures', and 'total_chunks_created' counts

    Raises:
        ValueError: If db is None, both file lists and directories provided, neither provided,
                   paths are not directories, or file counts don't match
        FileNotFoundError: If specified directories don't exist
        PermissionError: If directory access is denied
        OSError: For file system errors when accessing directories or files
        UnicodeDecodeError: If file reading encounters encoding errors (propagated from chunk_file)
        json.JSONDecodeError: If metadata JSON is malformed (propagated from chunk_file)
        Exception: For AI chunking, embedding, or database errors (propagated from chunk_file)
    """
    logger.info("Starting batch chunking operation", extra={
        "extra_fields": {
            "markdown_files_list_count": len(markdown_files_list) if markdown_files_list else None,
            "metadata_files_list_count": len(metadata_files_list) if metadata_files_list else None,
            "markdown_files_dir": str(markdown_files_dir) if markdown_files_dir else None,
            "metadata_files_dir": str(metadata_files_dir) if metadata_files_dir else None
        }
    })

    # Validate database session
    if db is None:
        logger.error("Database session cannot be None")
        raise ValueError("db session cannot be None")

    # Validate input mode
    list_mode = markdown_files_list is not None or metadata_files_list is not None
    dir_mode = markdown_files_dir is not None or metadata_files_dir is not None

    if list_mode and dir_mode:
        logger.error("Both file lists and directory paths provided")
        raise ValueError("Provide either file lists OR directory paths, not both.")

    if not list_mode and not dir_mode:
        logger.error("Neither file lists nor directory paths provided")
        raise ValueError("You must provide either file lists OR directory paths.")

    # Directory mode - discover files
    if dir_mode:
        # Convert and validate markdown directory
        if not isinstance(markdown_files_dir, Path):
            markdown_files_dir = Path(markdown_files_dir)

        if not markdown_files_dir.exists():
            logger.error(
                "Markdown directory does not exist",
                extra={
                    'extra_fields': {
                        "path": str(markdown_files_dir)
                    }
                }
            )
            raise FileNotFoundError(f"Directory does not exist: {markdown_files_dir}")

        if not markdown_files_dir.is_dir():
            logger.error(
                "Markdown path is not a directory",
                extra={
                    'extra_fields': {
                        "path": str(markdown_files_dir)
                    }
                }
            )
            raise ValueError(f"Path is not a directory: {markdown_files_dir}")

        # Convert and validate metadata directory
        if not isinstance(metadata_files_dir, Path):
            metadata_files_dir = Path(metadata_files_dir)

        if not metadata_files_dir.exists():
            logger.error(
                "Metadata directory does not exist",
                extra={
                    'extra_fields': {
                        "path": str(metadata_files_dir)
                    }
                }
            )
            raise FileNotFoundError(f"Directory does not exist: {metadata_files_dir}")

        if not metadata_files_dir.is_dir():
            logger.error(
                "Metadata path is not a directory",
                extra={
                    'extra_fields': {
                        "path": str(metadata_files_dir)
                    }
                }
            )
            raise ValueError(f"Path is not a directory: {metadata_files_dir}")

        # Discover files
        try:
            markdown_files_list = sorted([
                markdown_files_dir / f
                for f in os.listdir(markdown_files_dir)
                if (markdown_files_dir / f).is_file() and f.endswith(".md")
            ])

            metadata_files_list = sorted([
                metadata_files_dir / f
                for f in os.listdir(metadata_files_dir)
                if (metadata_files_dir / f).is_file() and f.endswith(".json")
            ])

        except (PermissionError, OSError) as e:
            logger.error("Failed to access directory", exc_info=True, extra={
                "extra_fields": {
                    "markdown_dir": str(markdown_files_dir),
                    "metadata_dir": str(metadata_files_dir),
                    "error_type": type(e).__name__
                }
            })
            raise

    # Validate file lists match
    if len(markdown_files_list) != len(metadata_files_list):
        logger.error("Mismatch between markdown and metadata file counts", extra={
            "extra_fields": {
                "markdown_count": len(markdown_files_list),
                "metadata_count": len(metadata_files_list)
            }
        })
        raise ValueError(
            f"File count mismatch: {len(markdown_files_list)} markdown files "
            f"vs {len(metadata_files_list)} metadata files"
        )

    if len(markdown_files_list) == 0:
        logger.warning("No files to process", extra={
            "extra_fields": {
                "markdown_dir": str(markdown_files_dir) if markdown_files_dir else None,
                "metadata_dir": str(metadata_files_dir) if metadata_files_dir else None
            }
        })
        return {
            "success": 0,
            "partial_failures": 0,
            "complete_failures": 0
        }

    # Create tasks for parallel processing
    start_time = time.time()

    tasks = [
        chunk_file(markdown_file, metadata_file, db)
        for markdown_file, metadata_file
        in zip(markdown_files_list, metadata_files_list)
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Analyze results
    complete_failures = 0
    partial_failures = 0
    success = 0
    total_chunks_created = 0

    for idx, result in enumerate(results):
        if isinstance(result, Exception):
            complete_failures += 1
            logger.error(f"Complete failure for file {idx + 1}", extra={
                "extra_fields": {
                    "file_index": idx,
                    "markdown_file": str(markdown_files_list[idx]),
                    "metadata_file": str(metadata_files_list[idx]),
                    "error_type": type(result).__name__
                }
            })
        else:
            # Count successful chunks (total_chunks - failed_chunks)
            successful_chunks = result['total_chunks'] - result['failed_chunks']
            total_chunks_created += successful_chunks

            if result['failed_chunks'] == 0:
                success += 1
            else:
                partial_failures += 1

    processing_duration = time.time() - start_time

    response = {
        "success": success,
        "partial_failures": partial_failures,
        "complete_failures": complete_failures,
        "total_chunks_created": total_chunks_created,
    }

    logger.info("Batch chunking completed", extra={
        "extra_fields": {
            **response,
            "total_files": len(markdown_files_list),
            "duration_seconds": round(processing_duration, 4),
            "success_rate": f"{(success / len(markdown_files_list) * 100):.1f}%" if markdown_files_list else "0%"
        }
    })

    return response


