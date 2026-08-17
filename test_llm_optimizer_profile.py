import json
import unittest
from unittest.mock import patch

import llm_optimizer
from config import MistralSettings, OptimizerConfig, RetrievalConfig
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

        with patch.object(llm_optimizer, "MistralClient", _FakeClient):
            proposed, _ = llm_optimizer.propose_next_config_llm(
                current_config=RetrievalConfig(),
                scores={},
                opt_config=OptimizerConfig(),
                mistral_settings=MistralSettings(api_key="test"),
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
