from __future__ import annotations

from app.config import Settings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from provider import get_chat_model


class LLMAdapter:
    """Adapter pattern: wraps provider initialization.

    - Custom provider: tự xử lý (cho phép API key rỗng, dùng ChatOpenAI)
    - Các provider khác (gemini/openai/openrouter/ollama): pass-through sang get_chat_model()
    """

    @staticmethod
    def build(settings: Settings) -> BaseChatModel:
        provider = settings.provider

        if provider == "custom":
            model_name = settings.custom_llm_model or settings.model
            api_key = settings.custom_llm_api_key or "sk-no-auth-required"
            base_url = settings.custom_llm_base_url
            if not base_url:
                raise ValueError("CUSTOM_LLM_BASE_URL is required for provider=custom")
            return ChatOpenAI(
                model=model_name,
                api_key=api_key,
                base_url=base_url,
                temperature=settings.temperature,
            )

        return get_chat_model(settings)
