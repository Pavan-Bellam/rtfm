from pathlib import Path

from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = ConfigDict(
        env_file = '.env',
        extra="ignore"
    )

    ENVIRONMENT: str
    FIRECRAWL_API: str
    LOG_LEVEL: str | None = "INFO"
    JSON_LOGS: bool | None = True
    LOG_FILE: bool | None= True
    OPENAI_API_KEY: str | None = None
    DEEPSEEK_API_KEY: str | None = None
    CHUNK_SIZE: int | None = 500
    SPLIT_SIZE: int | None = 20_000

    DATABASE_URL: str
    DB_POOL_SIZE: int | None = 5
    DB_MAX_OVERFLOW: int | None = 10

    @property
    def root_dir(self):
        return Path(__file__).resolve().parent.parent.parent

    @property
    def raw_data_storage_url(self):
        if self.ENVIRONMENT=='dev':
            return self.root_dir / "raw_data"

    @property
    def database_url_sync(self):
        db_url = self.DATABASE_URL.replace('asyncpg','psycopg2')
        return db_url


settings = Settings()
