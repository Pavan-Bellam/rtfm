
from pydantic import BaseModel, Field


#chunking Agent
class Chunk(BaseModel):
    """
    Represents a semantic chunk of documentation with line boundaries and context.
    """
    start_line_number: int = Field(..., description="The line number where this chunk starts (inclusive)")
    end_line_number: int = Field(..., description="The line number where this chunk ends (inclusive)")
    context: str = Field(..., description="A concise 1-2 sentence summary explaining what this chunk covers and its relevance")

class ChunkingAgentOutput(BaseModel):
    """
    Response schema of Agent that chunks the input data
    A list of Chunk objects
    """

    chunks: list[Chunk] = Field(..., description="List of chunks derived from the input text")


class ChunkingAgentState(BaseModel):
    input_text: str | None = Field(default=None, description="Input text that needs to be chunked by the agent")
    chunks: list[Chunk] | None = Field(default=None, description="List of chunks derived from the input text")
    messages: list | None = Field(default_factory=list, description="Conversation history for the agent")
    validation_errors: list[dict] | None = Field(default_factory=list, description="List of validation errors found in chunks")
    retry_count: int | None = Field(default=0, description="Number of retry attempts for fixing chunks")
