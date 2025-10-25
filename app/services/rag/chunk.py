from langchain_openai import ChatOpenAI

from app.core.logging import get_logger
from app.core.settings import settings

logger = get_logger(__name__)


llm = ChatOpenAI(api_key=settings.OPENAI_API_KEY)

file_loc = ""

