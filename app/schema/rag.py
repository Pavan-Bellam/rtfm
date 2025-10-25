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
