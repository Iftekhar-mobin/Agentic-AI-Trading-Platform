"""LLM client adapters and model catalogs."""

from atp.infrastructure.llm.anthropic_client import AnthropicLLMClient
from atp.infrastructure.llm.catalogs import (
    AnthropicModelCatalog,
    OllamaModelCatalog,
    OpenRouterModelCatalog,
)
from atp.infrastructure.llm.ollama_client import OllamaLLMClient
from atp.infrastructure.llm.openrouter_client import OpenRouterLLMClient
from atp.infrastructure.llm.router import LLMRouter

__all__ = [
    "AnthropicLLMClient",
    "AnthropicModelCatalog",
    "LLMRouter",
    "OllamaLLMClient",
    "OllamaModelCatalog",
    "OpenRouterLLMClient",
    "OpenRouterModelCatalog",
]
