from sqlalchemy.ext.asyncio import ( 
    create_async_engine,
    AsyncSession, 
    async_sessionmaker,
    AsyncEngine
)
from sqlalchemy.orm import declarative_base
from typing import AsyncGenerator

from app.core.settings import settings

engine: AsyncEngine = create_async_engine(
    settings.DATABASE_URL,
    echo= True if settings.ENVIRONMENT=='dev' else False,  # Only log SQL in debug mode
    pool_size=settings.DB_POOL_SIZE,  # Default: 5
    max_overflow=settings.DB_MAX_OVERFLOW,  # Default: 10
    pool_pre_ping=True,  # Verify connections before using
    pool_recycle=3600,  # Recycle connections after 1 hour
)

# Session factory
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

Base = declarative_base()

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    # TODO Handle exceptions correctly
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()