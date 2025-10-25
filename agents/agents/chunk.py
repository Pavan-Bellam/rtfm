import asyncio
from typing import Any, Optional, Dict, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel
from app.core.logging import get_logger, setup_logging
from agents.agents.basemodel import AgentBaseModel
from agents.prompts.chunk import CHUNKING_AGENT_PROMPT, CHUNKING_FIX_PROMPT_TEMPLATE
from agents.schema.rag import ChunkingAgentOutput, ChunkingAgentState

from langgraph.graph import START, END, StateGraph
from langgraph.graph.state import CompiledStateGraph

logger = get_logger(__name__)


class ChunkingAgent(AgentBaseModel):
    """
    Agent for intelligently chunking documents into smaller, semantic units.
    Uses LLM to understand document structure and create meaningful chunks.
    """

    def __init__(
        self,
        provider: Literal["openai", "deepseek"] = "openai",
        model: str = "gpt-4o-mini",
        model_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the chunking agent.

        Args:
            provider: LLM provider to use (default: "openai")
            model: Model identifier (default: "gpt-4o-mini")
            model_config: Optional model configuration (temperature, max_tokens, etc.)
        """

        # Call parent constructor
        super().__init__(
            name="chunking_agent",
            description="Intelligently chunks documents into semantic units with context",
            provider=provider,
            model=model,
            system_prompt=CHUNKING_AGENT_PROMPT,
            output_model=ChunkingAgentOutput,
            keep_history=False,
            model_config=model_config
        )

        logger.info(f"ChunkingAgent initialized with {provider}/{model}")

    def _extract_line_range(self, text: str, start_line: int, end_line: int) -> str:
        """
        Extract a specific line range from text with line numbers.

        Args:
            text: The full input text
            start_line: Starting line number (inclusive, 1-indexed)
            end_line: Ending line number (inclusive, 1-indexed)

        Returns:
            Extracted text with line numbers formatted as #N: content
        """
        lines = text.split('\n')
        extracted = []
        for i in range(start_line - 1, min(end_line, len(lines))):
            extracted.append(f"#{i+1}: {lines[i]}")
        return '\n'.join(extracted)

    async def _call_model(self, state: ChunkingAgentState) -> ChunkingAgentState:
        """
        Call the LLM to generate chunks from the input text using message history.
        """
        if state.input_text is None:
            logger.error("input text is empty while calling chunking agent")
            raise ValueError("Input text is empty while calling chunking agent")

        # Build messages with history
        if not state.messages:
            # First call - initialize with system prompt and input
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=state.input_text)
            ]
        else:
            # Subsequent calls - use accumulated history
            messages = state.messages

        # Use structured output
        llm_with_structure = self.llm.with_structured_output(self.output_model)
        result = await llm_with_structure.ainvoke(messages)

        # Store chunks and update message history
        state.chunks = result.chunks if hasattr(result, 'chunks') else []
        state.messages = messages  # Keep history for potential fixes

        logger.info(f"Generated {len(state.chunks)} chunks")

        return state

    def _validate(self, state: ChunkingAgentState) -> ChunkingAgentState:
        """
        Validate chunks for continuity and complete coverage.
        Sorts chunks, validates, and removes faulty chunks.
        """
        if not state.chunks:
            state.validation_errors = [{"type": "no_chunks", "message": "No chunks generated"}]
            return state

        # Sort chunks by start_line_number first
        state.chunks.sort(key=lambda c: c.start_line_number)
        logger.debug(f"Sorted {len(state.chunks)} chunks by start_line_number")

        # Get total lines from input_text
        total_lines = len(state.input_text.split('\n'))
        issues = []
        chunks_to_remove = set()  # Track indices of chunks to delete

        logger.debug(f"Validating {len(state.chunks)} chunks against {total_lines} total lines")

        # Check: First chunk starts at 1
        if state.chunks[0].start_line_number != 1:
            issues.append({
                "type": "missing_start",
                "message": f"First chunk starts at line {state.chunks[0].start_line_number}, should start at 1",
                "missing_range": (1, state.chunks[0].end_line_number)
            })
            chunks_to_remove.add(0)  # Delete first chunk, will be replaced

        # Check: Continuity between chunks
        for i in range(1, len(state.chunks)):
            expected_start = state.chunks[i-1].end_line_number + 1
            actual_start = state.chunks[i].start_line_number

            if actual_start != expected_start:
                gap_or_overlap = "gap" if actual_start > expected_start else "overlap"

                if gap_or_overlap == "gap":
                    # There's a gap - need full region context
                    issues.append({
                        "type": "gap",
                        "message": f"Gap between chunk {i-1} and {i}: "
                                  f"chunk {i-1} ends at {state.chunks[i-1].end_line_number}, "
                                  f"chunk {i} starts at {actual_start} (missing lines {expected_start}-{actual_start-1})",
                        "missing_range": (state.chunks[i-1].start_line_number, state.chunks[i].end_line_number)
                    })
                    chunks_to_remove.add(i-1)
                    chunks_to_remove.add(i)
                else:
                    # Overlap - delete the overlapping chunks
                    issues.append({
                        "type": "overlap",
                        "message": f"Overlap between chunk {i-1} and {i}: "
                                  f"chunk {i-1} ends at {state.chunks[i-1].end_line_number}, "
                                  f"chunk {i} starts at {actual_start}",
                        "missing_range": (state.chunks[i-1].start_line_number, state.chunks[i].end_line_number)
                    })
                    chunks_to_remove.add(i-1)
                    chunks_to_remove.add(i)

        # Check: Last chunk ends at total_lines (only if chunks remain)
        if len(state.chunks) > 0 and state.chunks[-1].end_line_number != total_lines:
            issues.append({
                "type": "missing_end",
                "message": f"Last chunk ends at line {state.chunks[-1].end_line_number}, should end at {total_lines}",
                "missing_range": (state.chunks[-1].start_line_number, total_lines)
            })
            chunks_to_remove.add(len(state.chunks) - 1)  # Delete last chunk, will be replaced

        # Remove faulty chunks (overlaps)
        if chunks_to_remove:
            state.chunks = [chunk for i, chunk in enumerate(state.chunks) if i not in chunks_to_remove]
            logger.info(f"Removed {len(chunks_to_remove)} overlapping chunks")

        state.validation_errors = issues

        if issues:
            logger.warning(f"Found {len(issues)} validation issues", extra={
                "extra_fields": {"issues": [i["type"] for i in issues]}
            })
        else:
            logger.info("Chunks validated successfully - no issues found")

        return state

    def _should_fix(self, state: ChunkingAgentState) -> str:
        """
        Conditional function to decide if chunks need fixing or are valid.

        Returns:
            "valid" if no errors or max retries reached, "fix" otherwise
        """
        if not state.validation_errors or len(state.validation_errors) == 0:
            return "valid"

        # Check retry count
        if state.retry_count >= 2:
            logger.warning(f"Max retries ({state.retry_count}) reached, accepting chunks with errors")
            return "valid"

        logger.info(f"Validation failed, attempting fix (retry {state.retry_count + 1}/2)")
        return "fix"

    async def _fix_gaps(self, state: ChunkingAgentState) -> ChunkingAgentState:
        """
        Generate chunks for missing/problematic regions.
        Each error is paired with its corresponding text region.
        """
        state.retry_count += 1

        logger.info(f"Fixing {len(state.validation_errors)} validation issues (attempt {state.retry_count})")

        # Build error-region pairs
        errors_with_regions = []

        for i, issue in enumerate(state.validation_errors, 1):
            error_msg = issue['message']

            # Extract text for the missing range
            if "missing_range" in issue:
                start_line, end_line = issue["missing_range"]
                region_text = self._extract_line_range(state.input_text, start_line, end_line)

                errors_with_regions.append(
                    f"ERROR {i}:\n{error_msg}\n\n"
                    f"TEXT FOR ERROR {i} (lines {start_line}-{end_line}):\n{region_text}"
                )

        separator = "\n\n" + "="*80 + "\n\n"
        combined_errors_regions = separator.join(errors_with_regions)

        # Build fix prompt
        fix_prompt = CHUNKING_FIX_PROMPT_TEMPLATE.format(
            errors_with_regions=combined_errors_regions
        )

        # Add fix prompt to message history
        state.messages.append(SystemMessage(content=fix_prompt))

        # Call LLM to generate chunks for all missing regions
        llm_with_structure = self.llm.with_structured_output(self.output_model)
        result = await llm_with_structure.ainvoke(state.messages)

        new_chunks = result.chunks if hasattr(result, 'chunks') else []

        logger.info(f"Received {len(new_chunks)} new chunks for {len(state.validation_errors)} regions")

        # Add new chunks to existing chunks (validation will sort next)
        state.chunks.extend(new_chunks)
        state.validation_errors = []  # Clear errors for re-validation

        logger.info(f"Total chunks after fix: {len(state.chunks)}")

        return state

    def _get_graph(self) -> CompiledStateGraph:
        """
        Build the execution graph for document chunking with validation loop.

        Flow:
        START → get_chunks → validate_chunks → [if valid] → END
                                              → [if invalid] → fix_gaps → validate_chunks (loop)

        Returns:
            Compiled LangGraph for chunking operations
        """
        graph = StateGraph(ChunkingAgentState)

        # Add nodes
        graph.add_node("get_chunks", self._call_model)
        graph.add_node("validate_chunks", self._validate)
        graph.add_node("fix_gaps", self._fix_gaps)

        # Add edges
        graph.add_edge(START, "get_chunks")
        graph.add_edge("get_chunks", "validate_chunks")

        # Conditional edge: valid → END, fix → fix_gaps
        graph.add_conditional_edges(
            "validate_chunks",
            self._should_fix,
            {
                "valid": END,
                "fix": "fix_gaps"
            }
        )

        # Loop back to validation after fixing
        graph.add_edge("fix_gaps", "validate_chunks")

        return graph.compile()

    async def run(self, document: str, **kwargs) -> Dict[str, Any]:
        """
        Chunk a document into semantic units with validation and auto-fixing.

        Args:
            document: The document text to chunk
            **kwargs: Additional parameters for chunking

        Returns:
            Dictionary containing:
            - chunks: List of Chunk objects
            - validation_errors: Any remaining validation errors (if max retries reached)
            - retry_count: Number of fix attempts made
        """
        logger.info(f"Starting document chunking", extra={
            "extra_fields": {
                "document_length": len(document),
                "total_lines": len(document.split('\n')),
                "agent": self.name
            }
        })

        # Create initial state and invoke graph
        state = ChunkingAgentState(input_text=document)
        result = await self.graph.ainvoke(state)

        # Log final results
        logger.info(f"Chunking complete", extra={
            "extra_fields": {
                "chunks_generated": len(result.get("chunks", [])),
                "retry_count": result.get("retry_count", 0),
                "has_errors": len(result.get("validation_errors", [])) > 0
            }
        })

        return {
            "chunks": result.get("chunks", []),
            "validation_errors": result.get("validation_errors", []),
            "retry_count": result.get("retry_count", 0)
        }


async def main():
    # Setup logging first
    setup_logging(
        log_level="INFO",
        json_logs=False,  # Set to False for readable console output during testing
        log_file=True
    )

    logger.info("Starting chunking agent test")

    chunking_agent = ChunkingAgent(
        provider='openai',
        model='gpt-5-nano'
    )

    file_path = 'D:/work/rtfm/raw_data/terraform-aws/markdown/0000_53ecb043cc47.md'
    logger.info(f"Reading file: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = f.read()

    result = await chunking_agent.run(document=data)

    logger.info("Writing results to file")

    # Convert Chunk objects to dicts
    result["chunks"] = [chunk.model_dump() for chunk in result["chunks"]]

    with open('chunking_agent_result.json', "w", encoding='utf-8') as f:
        import json
        json.dump(result["chunks"], f, indent=2)

    logger.info("Chunking complete!")

if __name__=="__main__":
    asyncio.run(main())    
