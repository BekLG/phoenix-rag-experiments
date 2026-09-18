"""Compatibility shims for optional integrations imported by Ragas."""

from __future__ import annotations

import sys
import types


def install_ragas_compat() -> None:
    """Provide the removed Vertex AI module that Ragas imports eagerly.

    Phoenix uses Mistral for every model call. Ragas 0.3.9 nevertheless imports
    ``ChatVertexAI`` at package import time, while recent langchain-community
    releases moved that integration out of the package.
    """
    vertex_module = "langchain_community.chat_models.vertexai"
    if vertex_module not in sys.modules:
        module = types.ModuleType(vertex_module)

        class ChatVertexAI:
            def __init__(self, *args, **kwargs):
                raise ImportError(
                    "ChatVertexAI is not used by Phoenix RAG; install "
                    "langchain-google-vertexai only if a Ragas integration needs it."
                )

        module.ChatVertexAI = ChatVertexAI
        sys.modules[vertex_module] = module

    mistral_module = "mistralai.async_client"
    if mistral_module not in sys.modules:
        sys.modules[mistral_module] = types.ModuleType(mistral_module)
