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

    @property
    def ROOT_DIR(self):
        return Path(__file__).resolve().parent.parent.parent

    @property
    def RAW_DATA_STORAGE_URL(self):
        if self.ENVIRONMENT=='dev':
            return self.ROOT_DIR / "raw_data"


settings = Settings()