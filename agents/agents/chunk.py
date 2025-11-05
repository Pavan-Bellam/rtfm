from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents.agents.basemodel import AgentBaseModel
from agents.prompts.chunk import CHUNKING_AGENT_PROMPT, CHUNKING_FIX_PROMPT_TEMPLATE
from agents.schema.rag import ChunkingAgentOutput, ChunkingAgentState
from app.core.logging import get_logger
from app.services.rag.utils import extract_line_range

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
        model_config: dict[str, Any] | None = None
    ):
        """
        Initialize the chunking agent.

        Args:
            provider: LLM provider to use (default: "openai")
            model: Model identifier (default: "gpt-4o-mini")
            model_config: Optional model configuration (temperature, max_tokens, etc.)

        Raises:
            Exception: If agent initialization fails
        """
        try:
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

            logger.info("ChunkingAgent initialized successfully", extra={
                "extra_fields": {
                    "provider": provider,
                    "model": model
                }
            })

        except Exception as e:
            logger.error("Failed to initialize ChunkingAgent", exc_info=True, extra={
                "extra_fields": {
                    "provider": provider,
                    "model": model,
                    "error_type": type(e).__name__
                }
            })
            raise

    async def _call_model(self, state: ChunkingAgentState) -> ChunkingAgentState:
        """
        Call the LLM to generate chunks from the input text using message history.

        Args:
            state: Current chunking agent state with input text

        Returns:
            Updated state with generated chunks

        Raises:
            ValueError: If input text is None or empty
            Exception: If LLM invocation fails
        """
        if state.input_text is None or not state.input_text.strip():
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
        if not hasattr(result, 'chunks'):
            logger.error("LLM result missing 'chunks' attribute", extra={
                "extra_fields": {
                    "result_type": type(result).__name__
                }
            })
            state.chunks = []
        else:
            state.chunks = result.chunks

        state.messages = messages  # Keep history for potential fixes

        return state

    def _validate(self, state: ChunkingAgentState) -> ChunkingAgentState:
        """
        Validate chunks for continuity and complete coverage.
        Sorts chunks, validates, and removes faulty chunks.

        Args:
            state: Current chunking agent state with chunks to validate

        Returns:
            Updated state with validation_errors populated

        Raises:
            ValueError: If input_text is missing from state
        """
        # Validate input_text exists
        if state.input_text is None or not state.input_text.strip():
            raise ValueError("Input text is missing during validation")

        if not state.chunks:
            state.validation_errors = [{"type": "no_chunks", "message": "No chunks generated"}]
            return state

        # Sort chunks by start_line_number first
        state.chunks.sort(key=lambda c: c.start_line_number)

        # Get total lines from input_text
        total_lines = len(state.input_text.split('\n'))
        issues = []
        chunks_to_remove = set()  # Track indices of chunks to delete

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

        # Remove faulty chunks
        if chunks_to_remove:
            state.chunks = [chunk for i, chunk in enumerate(state.chunks) if i not in chunks_to_remove]

        state.validation_errors = issues

        return state

    def _should_fix(self, state: ChunkingAgentState) -> str:
        """
        Conditional function to decide if chunks need fixing or are valid.

        Args:
            state: Current chunking agent state with validation results

        Returns:
            "valid" if no errors or max retries reached, "fix" otherwise
        """
        if not state.validation_errors or len(state.validation_errors) == 0:
            return "valid"

        # Check retry count
        if state.retry_count >= 2:
            return "valid"

        return "fix"

    async def _fix_gaps(self, state: ChunkingAgentState) -> ChunkingAgentState:
        """
        Generate chunks for missing/problematic regions.
        Each error is paired with its corresponding text region.

        Args:
            state: Current chunking agent state with validation errors

        Returns:
            Updated state with new chunks added

        Raises:
            ValueError: If input_text is missing from state
            Exception: If LLM invocation fails
        """
        state.retry_count += 1

        # Validate input_text exists
        if state.input_text is None or not state.input_text.strip():
            raise ValueError("Input text is missing during gap fixing")

        # Build error-region pairs
        errors_with_regions = []

        for i, issue in enumerate(state.validation_errors, 1):
            error_msg = issue['message']

            # Extract text for the missing range
            if "missing_range" in issue:
                start_line, end_line = issue["missing_range"]

                # Use the imported extract_line_range function
                region_text = extract_line_range(state.input_text, start_line, end_line)

                errors_with_regions.append(
                    f"ERROR {i}:\n{error_msg}\n\n"
                    f"TEXT FOR ERROR {i} (lines {start_line}-{end_line}):\n{region_text}"
                )

        if not errors_with_regions:
            return state

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

        if not hasattr(result, 'chunks'):
            logger.error("LLM result missing 'chunks' attribute during gap fixing", extra={
                "extra_fields": {
                    "result_type": type(result).__name__
                }
            })
            new_chunks = []
        else:
            new_chunks = result.chunks

        # Add new chunks to existing chunks (validation will sort next)
        state.chunks.extend(new_chunks)
        state.validation_errors = []  # Clear errors for re-validation

        return state

    def _get_graph(self) -> CompiledStateGraph:
        """
        Build the execution graph for document chunking with validation loop.

        Flow:
        START → get_chunks → validate_chunks → [if valid] → END
                                              → [if invalid] → fix_gaps → validate_chunks (loop)

        Returns:
            Compiled LangGraph for chunking operations

        Raises:
            Exception: If graph compilation fails
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

        compiled_graph = graph.compile()

        return compiled_graph

    async def run(self, document: str, **kwargs) -> dict[str, Any]:
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

        Raises:
            ValueError: If document is None or empty
            Exception: If graph execution fails
        """
        # Validate input
        if document is None or not document.strip():
            raise ValueError("Document cannot be empty or None")

        # Create initial state and invoke graph
        state = ChunkingAgentState(input_text=document)
        result = await self.graph.ainvoke(state)

        return {
            "chunks": result.get("chunks", []),
            "validation_errors": result.get("validation_errors", []),
            "retry_count": result.get("retry_count", 0)
        }