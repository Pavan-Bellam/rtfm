from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, bindparam, Integer
from pgvector.sqlalchemy import Vector

from app.core.db import get_db
from app.core.logging import get_logger
from app.core.rag_config import openai_embeddings
from typing import List
import asyncio

logger = get_logger(__name__)

async def get_top_k_similar_embedings(embeding: List[int], db: AsyncSession, top_k: int=5) -> List[dict[str,str]]:
    logger.info(f"Searching for top {top_k} similar embeddings")

    query = text("""
        SELECT id, content, context,
               1 - (embedding <=> :emb) as similarity
        FROM chunks
        ORDER BY embedding <=> :emb
        LIMIT :k
    """).bindparams(
        bindparam('emb', type_=Vector(1536)),
        bindparam('k', type_=Integer)
    )

    logger.debug("Executing similarity search query")
    result = await db.execute(
        query, {"emb": embeding, "k": top_k}
    )

    results = [dict(row._mapping) for row in result]
    logger.info(f"Found {len(results)} similar chunks")

    return results


async def retrive(query: str, db: AsyncSession, top_k: int=5) -> List[dict[str,str]]:
    logger.info(f"Starting retrieval for query: '{query}' (top_k={top_k})")

    logger.debug("Generating query embedding")
    embedding =  openai_embeddings.embed_query(query)
    logger.debug(f"Generated embedding with dimension: {len(embedding)}")

    results = await get_top_k_similar_embedings(embedding, db, top_k)

    logger.info(f"Retrieval completed, returning {len(results)} results")
    return results

