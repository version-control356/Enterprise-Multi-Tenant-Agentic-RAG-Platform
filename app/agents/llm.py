from functools import lru_cache
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq
from pydantic import SecretStr
from app.config import settings


@lru_cache(maxsize=4)
def get_llm_client(temperature: float = 0.2, max_tokens: int = 1500) -> BaseChatModel:
    """Instantiates the high-speed Groq Cloud LLM engine with instance caching."""
    return ChatGroq(
        api_key=SecretStr(settings.GROQ_API_KEY),
        model=settings.GROQ_MODEL,
        temperature=temperature,
        max_retries=2,
        max_tokens=max_tokens,
        timeout=30.0,
        streaming=True,
    )