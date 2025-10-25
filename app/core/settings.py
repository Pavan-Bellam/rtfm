from pathlib import Path
from typing import Optional
from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = ConfigDict(
        env_file = '.env',
        extra="ignore"
    )

    ENVIRONMENT: str
    FIRECRAWL_API: str
    LOG_LEVEL: Optional[str] = "INFO"
    JSON_LOGS: Optional[bool] = True
    LOG_FILE: Optional[bool]= True
    OPENAI_API_KEY: Optional[str] = None
    DEEPSEEK_API_KEY: Optional[str] = None
    CHUNK_SIZE: Optional[int] = 500
    SPLIT_SIZE: Optional[int] = 20_000

    DATABASE_URL: str
    DB_POOL_SIZE: Optional[int] = 5
    DB_MAX_OVERFLOW: Optional[int] = 10

    @property
    def ROOT_DIR(self):
        return Path(__file__).resolve().parent.parent.parent

    @property
    def RAW_DATA_STORAGE_URL(self):
        if self.ENVIRONMENT=='dev':
            return self.ROOT_DIR / "raw_data"
    
    @property
    def DATABASE_URL_SYNC(self):
        db_url = self.DATABASE_URL.replace('asyncpg','psycopg2')
        return db_url


settings = Settings()