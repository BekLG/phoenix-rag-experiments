import json
import sys
import types
import unittest
from unittest.mock import patch

try:
    from google import genai as _genai  # noqa: F401
except ImportError:
    google = types.ModuleType("google")
    genai = types.ModuleType("google.genai")
    genai.Client = object
    genai.types = types.SimpleNamespace(GenerateContentConfig=object)
    google.genai = genai
    sys.modules["google"] = google
    sys.modules["google.genai"] = genai

import llm_optimizer
from config import GeminiSettings, OptimizerConfig, RetrievalConfig
from document_profile import DocumentProfile


class _FakeClient:
    response: str = ""
    last_messages = None

    def __init__(self, settings):
        self.settings = settings

    def chat(self, messages, **kwargs):
        type(self).last_messages = messages
        return type(self).response


class OptimizerProfileTests(unittest.TestCase):
    def test_profile_does_not_bypass_clamping(self):
        _FakeClient.response = json.dumps(
            {
                "chunk_size": 99999,
                "chunk_overlap": 99999,
                "top_k": 999,
                "retriever_type": "similarity",
                "similarity_threshold": 99,
                "prompt_template": "Context: {context}\nQuestion: {question}",
                "reasoning": "test out-of-bounds values",
            }
        )
        profile = DocumentProfile(
            pages=1,
            characters=200,
            estimated_tokens=50,
            sections=None,
            median_chars_per_page=200.0,
            min_chars_per_page=200,
            max_chars_per_page=200,
            doc_type="unknown",
            table_heavy=False,
            list_heavy=False,
        )

        with patch.object(llm_optimizer, "GeminiClient", _FakeClient):
            proposed, _ = llm_optimizer.propose_next_config_llm(
                current_config=RetrievalConfig(),
                scores={},
                opt_config=OptimizerConfig(),
                gemini_settings=GeminiSettings(api_key="test"),
                history=[],
                document_summary="Short document.",
                document_profile=profile,
            )

        self.assertEqual(proposed.chunk_size, 1500)
        self.assertEqual(proposed.chunk_overlap, 1499)
        self.assertEqual(proposed.top_k, 10)
        self.assertEqual(proposed.similarity_threshold, 0.75)
        user_message = _FakeClient.last_messages[1]["content"]
        self.assertIn("DOCUMENT PROFILE:", user_message)
        self.assertIn("estimated_chunk_count=1", user_message)


if __name__ == "__main__":
    unittest.main()
