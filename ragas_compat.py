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
    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules:
        return

    module = types.ModuleType(module_name)

    class ChatVertexAI:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "ChatVertexAI is not used by Phoenix RAG; install "
                "langchain-google-vertexai only if a Ragas integration needs it."
            )

    module.ChatVertexAI = ChatVertexAI
    sys.modules[module_name] = module
