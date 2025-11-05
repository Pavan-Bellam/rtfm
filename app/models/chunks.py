from sqlalchemy import Column, String, Integer, func, TIMESTAMP, Index
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector
from app.core.db import Base
from app.core.rag_config import embed_dim
from uuid import uuid4

class Chunk(Base):
    
    __tablename__ = "chunks"

    id = Column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid4
    )
    tool_name = Column(String, nullable=False)
    url = Column(String, nullable=False)
    content = Column(String, nullable=False)
    context = Column(String, nullable=False)
    embedding = Column(Vector(embed_dim))
    created_at = Column(
        TIMESTAMP(timezone=True),
        nullable=False,
        server_default=func.now()
    )

    __table_args__ = (
        Index(
            "chunks_embedding_hnsw_idx",
            embedding,
            postgresql_using='hnsw',
            postgresql_ops={'embedding': 'vector_cosine_ops'}
        ),
    )
