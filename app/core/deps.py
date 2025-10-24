from firecrawl import AsyncFirecrawl
from app.core.settings import settings

firecrawl = AsyncFirecrawl(api_key=settings.FIRECRAWL_API)