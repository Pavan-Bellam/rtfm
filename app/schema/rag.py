from pydantic import BaseModel, Field, HttpUrl, field_validator


class ScrapeRequest(BaseModel):
    """
    Request schema for web scraping operation.
    Defines the target URL, crawl limit, and tool identifier for organizing scraped data.
    """
    tool_name: str = Field(..., min_length=1, description="Name of the tool/project for organizing scraped data")
    url: HttpUrl = Field(..., description="Starting URL for web crawling")
    limit: int = Field(..., gt=0, le=1000, description="Maximum number of pages to scrape")

    @field_validator('tool_name')
    @classmethod
    def validate_tool_name(cls, v: str) -> str:
        """
        Validate tool_name to ensure it's filesystem-safe.
        Prevents directory traversal and invalid characters in folder names.
        """
        if not v or not v.strip():
            raise ValueError("tool_name cannot be empty or whitespace")

        invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
        if any(char in v for char in invalid_chars):
            raise ValueError(f"tool_name contains invalid characters: {invalid_chars}")

        return v.strip()


class ScrapeResponse(BaseModel):
    """
    Response schema for web scraping operation.
    Returns the directory paths where markdown files and metadata are stored.
    """
    markdown_dir: str = Field(..., description="Directory path containing scraped markdown files")
    metadata_dir: str = Field(..., description="Directory path containing metadata JSON files")


class IngestionRequest(BaseModel):
    """
    Request schema for complete ingestion pipeline.
    Combines scraping, chunking, embedding, and storage in one operation.
    """
    tool_name: str = Field(..., min_length=1, description="Name of the tool/project for organizing data")
    url: HttpUrl = Field(..., description="Starting URL for web crawling")
    limit: int = Field(..., gt=0, le=10000, description="Maximum number of pages to scrape and ingest")

    @field_validator('tool_name')
    @classmethod
    def validate_tool_name(cls, v: str) -> str:
        """
        Validate tool_name to ensure it's filesystem-safe.
        Prevents directory traversal and invalid characters in folder names.
        """
        if not v or not v.strip():
            raise ValueError("tool_name cannot be empty or whitespace")

        invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
        if any(char in v for char in invalid_chars):
            raise ValueError(f"tool_name contains invalid characters: {invalid_chars}")

        return v.strip()


class IngestionResponse(BaseModel):
    """
    Response schema for ingestion pipeline operation.
    Returns statistics about the scraping, chunking, and embedding process.
    """
    message: str = Field(..., description="Success or status message")
    pages_scraped: int = Field(..., description="Number of web pages successfully scraped")
    files_processed: int = Field(..., description="Total number of files processed for chunking")
    files_success: int = Field(..., description="Number of files successfully chunked and embedded")
    files_partial_failures: int = Field(..., description="Number of files with some chunks failing")
    files_complete_failures: int = Field(..., description="Number of files that completely failed to process")
    total_chunks_created: int = Field(..., description="Total number of chunks created and stored in database")
    markdown_dir: str = Field(..., description="Directory path containing scraped markdown files")
    metadata_dir: str = Field(..., description="Directory path containing metadata JSON files")
