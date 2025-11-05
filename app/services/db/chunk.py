from typing import List

from sqlalchemy import insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.chunks import Chunk

logger = get_logger(__name__)


async def create_chunk(
    tool_name: str,
    url: str,
    content: str,
    context: str,
    embedding: List[float],
    db: AsyncSession
) -> str:
    """
    Create a row in chunks table.
    Args:
        tool_name: Name of the tool that chunk belongs to
        url: URL from which the chunk is picked
        content: The actual content of the chunk
        context: 1-2 line summary of the content
        embedding: Embedding vector for the chunk
        db: AsyncSession to the database

    Returns:
        str: The ID of the created chunk

    Raises:
        ValueError: If any required parameter is empty or invalid
        SQLAlchemyError: If database operation fails
        Exception: For unexpected errors during chunk creation
    """
    logger.debug("Creating new chunk in database", extra={
        "extra_fields": {
            "tool_name": tool_name,
            "url": url,
            "content_length": len(content) if content else 0,
            "context_length": len(context) if context else 0,
            "embedding_dimensions": len(embedding) if embedding else 0
        }
    })

    # Validate inputs
    if not tool_name or not tool_name.strip():
        logger.error("Invalid tool_name: cannot be empty")
        raise ValueError("tool_name cannot be empty")

    if not url or not url.strip():
        logger.error("Invalid url: cannot be empty", extra={
            "extra_fields": {"tool_name": tool_name}
        })
        raise ValueError("url cannot be empty")

    if not content or not content.strip():
        logger.error("Invalid content: cannot be empty", extra={
            "extra_fields": {"tool_name": tool_name, "url": url}
        })
        raise ValueError("content cannot be empty")

    if not context or not context.strip():
        logger.error("Invalid context: cannot be empty", extra={
            "extra_fields": {"tool_name": tool_name, "url": url}
        })
        raise ValueError("context cannot be empty")

    if not embedding or len(embedding) == 0:
        logger.error("Invalid embedding: cannot be empty", extra={
            "extra_fields": {"tool_name": tool_name, "url": url}
        })
        raise ValueError("embedding cannot be empty")

    if db is None:
        logger.error("Invalid db session: cannot be None")
        raise ValueError("db session cannot be None")

    try:
        stmt = insert(Chunk).values(
            tool_name=tool_name,
            url=url,
            content=content,
            context=context,
            embedding=embedding
        ).returning(Chunk.id)

        result = await db.execute(stmt)
        await db.commit()
        chunk_id = str(result.scalar_one())

        logger.debug("Chunk created successfully", extra={
            "extra_fields": {
                "chunk_id": chunk_id,
                "tool_name": tool_name,
                "url": url,
                "content_length": len(content),
                "embedding_dimensions": len(embedding)
            }
        })

        return chunk_id

    except SQLAlchemyError:
        await db.rollback()
        logger.error("Database error while creating chunk", exc_info=True, extra={
            "extra_fields": {
                "tool_name": tool_name,
                "url": url,
                "error_type": "SQLAlchemyError"
            }
        })
        raise

    except Exception as e:
        await db.rollback()
        logger.error("Unexpected error while creating chunk", exc_info=True, extra={
            "extra_fields": {
                "tool_name": tool_name,
                "url": url,
                "error_type": type(e).__name__
            }
        })
        raise    
