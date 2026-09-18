import json
import unittest

from phoenix_rag.config import OptimizerConfig, RetrievalConfig
from phoenix_rag.core.document_profile import DocumentProfile
from phoenix_rag.optimization import llm_optimizer


class _FakeOptimizer:
    """A minimal ChatProvider stand-in for the optimizer role.

    Records the messages it was asked to send and returns a canned reply. It
    carries no model: a provider is already bound to its role's model, so the
    proposer never passes one -- which is exactly why the old signature (a
    MistralSettings + a patched MistralClient) is gone.
    """

    response: str = ""
    last_messages = None

    def chat(self, messages, **kwargs):
        type(self).last_messages = messages
        return type(self).response


class OptimizerProfileTests(unittest.TestCase):
    def test_profile_does_not_bypass_clamping(self):
        _FakeOptimizer.response = json.dumps(
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

        proposed, _ = llm_optimizer.propose_next_config_llm(
            current_config=RetrievalConfig(),
            scores={},
            opt_config=OptimizerConfig(),
            optimizer=_FakeOptimizer(),
            history=[],
            document_summary="Short document.",
            document_profile=profile,
        )

        self.assertEqual(proposed.chunk_size, 1500)
        self.assertEqual(proposed.chunk_overlap, 1499)
        self.assertEqual(proposed.top_k, 10)
        self.assertEqual(proposed.similarity_threshold, 0.75)
        user_message = _FakeOptimizer.last_messages[1]["content"]
        self.assertIn("DOCUMENT PROFILE:", user_message)
        self.assertIn("estimated_chunk_count=1", user_message)


if __name__ == "__main__":
    unittest.main()
