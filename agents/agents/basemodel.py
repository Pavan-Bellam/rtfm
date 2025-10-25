from typing import Literal, Optional, Any, Dict, List, Type
from abc import ABC, abstractmethod
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from app.core.settings import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class AgentBaseModel(ABC):
    """
    Abstract base class for LangChain-based agents.
    Provides common functionality for initializing LLM providers, managing state, and building agent graphs.
    Subclasses must implement _get_state, _get_graph, and run methods.
    """

    def __init__(
        self,
        name: str,
        description: str,
        provider: Literal["openai", "deepseek"],
        model: str,
        system_prompt: str,
        output_model: BaseModel,
        keep_history: bool = False,
        model_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the agent with LLM configuration and agent-specific settings.

        Args:
            name: Identifier name for the agent
            description: Human-readable description of agent purpose
            provider: LLM provider to use ("openai" or "deepseek")
            model: Model identifier (e.g., "gpt-4", "deepseek-chat")
            system_prompt: System prompt to guide agent behavior
            output_model: Pydantic model for structured output
            keep_history: Whether to maintain conversation history
            model_config: Additional configuration for LLM (temperature, max_tokens, etc.)

        Raises:
            ValueError: If API key is not configured for the selected provider
        """
        self.name = name
        self.description = description
        self.provider = provider
        self.model = model
        self.system_prompt = system_prompt
        self.output_model = output_model
        self.keep_history = keep_history
        self.model_config = model_config or {}

        # Initialize LLM
        self.llm = self._get_llm(provider, model, self.model_config)

        # Initialize history if needed
        if self.keep_history:
            self.history: List[Dict[str, Any]] = []

        # Initialize graph (to be implemented by subclasses)
        self.graph = self._get_graph()

        logger.info(f"Agent initialized", extra={
            "extra_fields": {
                "agent_name": self.name,
                "provider": self.provider,
                "model": self.model,
                "keep_history": self.keep_history
            }
        })

    def _get_llm(
        self,
        provider: Literal["openai", "deepseek"],
        model: str,
        model_config: Dict[str, Any]
    ) -> ChatOpenAI | ChatDeepSeek:
        """
        Initialize and return the appropriate LLM client based on provider.

        Args:
            provider: LLM provider to use ("openai" or "deepseek")
            model: Model identifier
            model_config: Additional model configuration parameters

        Returns:
            Initialized LLM client instance

        Raises:
            ValueError: If API key is not configured for the selected provider
        """
        if provider == "openai":
            if not settings.OPENAI_API_KEY:
                logger.critical("OPENAI_API_KEY is not set in environment variables", extra={
                    "extra_fields": {"provider": provider, "model": model}
                })
                raise ValueError("OPENAI_API_KEY is not configured. Please set it in environment variables.")

            logger.debug(f"Initializing OpenAI LLM", extra={
                "extra_fields": {"model": model, "config": model_config}
            })
            return ChatOpenAI(
                model=model,
                api_key=settings.OPENAI_API_KEY,
                **model_config
            )

        elif provider == "deepseek":
            if not settings.DEEPSEEK_API_KEY:
                logger.critical("DEEPSEEK_API_KEY is not set in environment variables", extra={
                    "extra_fields": {"provider": provider, "model": model}
                })
                raise ValueError("DEEPSEEK_API_KEY is not configured. Please set it in environment variables.")

            logger.debug(f"Initializing DeepSeek LLM", extra={
                "extra_fields": {"model": model, "config": model_config}
            })
            return ChatDeepSeek(
                model=model,
                api_key=settings.DEEPSEEK_API_KEY,
                **model_config
            )

        else:
            logger.error(f"Unsupported provider: {provider}")
            raise ValueError(f"Unsupported provider: {provider}. Must be 'openai' or 'deepseek'.")



    @abstractmethod
    def _get_graph(self) -> CompiledStateGraph:
        """
        Build and return the agent's execution graph.
        Must be implemented by subclasses.

        Returns:
            Compiled graph object (typically a LangGraph CompiledGraph)
        """
        pass

    @abstractmethod
    async def run(self, *args, **kwargs) -> Any:
        """
        Execute the agent with given inputs.
        Must be implemented by subclasses.

        Args:
            *args: Positional arguments specific to agent implementation
            **kwargs: Keyword arguments specific to agent implementation

        Returns:
            Agent execution result (structure depends on implementation)
        """
        pass

    def clear_history(self) -> None:
        """
        Clear the conversation history if history tracking is enabled.
        """
        if self.keep_history:
            logger.info(f"Clearing history for agent", extra={
                "extra_fields": {
                    "agent_name": self.name,
                    "previous_history_length": len(self.history)
                }
            })
            self.history.clear()
        else:
            logger.warning(f"Attempted to clear history but keep_history is False", extra={
                "extra_fields": {"agent_name": self.name}
            })