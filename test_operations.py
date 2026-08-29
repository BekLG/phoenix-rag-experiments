"""
test_operations.py
==================
Offline tests for the shared operator layer -- specifically the configuration
editor, which is the part with real logic in it. Both front-ends drive these
functions, so a rule tested here is a rule enforced in the terminal AND the GUI.

Nothing here touches the Mistral API, the filesystem outside a temp dir, or FAISS.
"""

from __future__ import annotations

import unittest

import operations
from config import AppConfig


class EditableFieldsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = AppConfig()
        self.fields = {field_.path: field_ for field_ in operations.editable_fields(self.config)}

    def test_the_nested_config_tree_is_flattened(self):
        # These are the knobs the request named explicitly, so they have to be
        # reachable from the editor without a code change.
        for path in (
            "question_generation.questions_per_batch",
            "question_generation.batch_size_chars",
            "optimizer.max_iterations",
            "retrieval.chunk_size",
            "retrieval.chunk_overlap",
            "retrieval.top_k",
            "retrieval.prompt_template",
            "mistral.requests_per_minute",
            "source_document",
        ):
            self.assertIn(path, self.fields)

    def test_corpus_path_is_not_hand_editable(self):
        # Pointing it at a stale root silently changes which documents are
        # searched; it is managed by the corpus operations instead.
        self.assertNotIn("corpus_path", self.fields)

    def test_kinds_are_classified_from_values_and_overrides(self):
        self.assertEqual(self.fields["retrieval.chunk_size"].kind, "int")
        self.assertEqual(self.fields["optimizer.seed_from_profile"].kind, "bool")
        self.assertEqual(self.fields["retrieval.similarity_threshold"].kind, "float")
        self.assertEqual(self.fields["retrieval.retriever_type"].kind, "choice")
        self.assertEqual(self.fields["retrieval.prompt_template"].kind, "text")
        self.assertEqual(self.fields["optimizer.top_k_bounds"].kind, "int_pair")
        self.assertEqual(
            self.fields["optimizer.similarity_threshold_bounds"].kind, "float_pair"
        )
        self.assertEqual(self.fields["question_generation.question_types"].kind, "str_list")

    def test_a_float_field_holding_a_whole_number_is_still_a_float(self):
        # A JSON config with "similarity_threshold": 0 would otherwise be
        # classified int, and every later edit silently truncated.
        self.config.retrieval.similarity_threshold = 0
        fields = {f.path: f for f in operations.editable_fields(self.config)}
        self.assertEqual(fields["retrieval.similarity_threshold"].kind, "float")

    def test_the_api_key_is_masked_for_display(self):
        self.config.mistral.api_key = "sk-abcdefghijklmnopqrstuvwxyz"
        fields = {f.path: f for f in operations.editable_fields(self.config)}
        field_ = fields["mistral.api_key"]
        self.assertTrue(field_.secret)
        self.assertNotIn("efghijklmnopqrstuv", field_.display_value)

    def test_bool_is_classified_before_int(self):
        # bool IS an int in Python; misordering the checks would turn every
        # boolean into a number field that accepts 7.
        self.assertNotEqual(self.fields["optimizer.seed_from_profile"].kind, "int")


class CoercionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = AppConfig()

    def apply(self, path: str, raw):
        return operations.apply_field(self.config, path, raw)

    def test_ints_floats_and_bools(self):
        self.assertEqual(self.apply("optimizer.max_iterations", "12"), 12)
        self.assertEqual(self.config.optimizer.max_iterations, 12)
        self.assertEqual(self.apply("retrieval.similarity_threshold", "0.35"), 0.35)
        self.assertIs(self.apply("optimizer.seed_from_profile", "no"), False)
        self.assertIs(self.apply("optimizer.seed_from_profile", "YES"), True)

    def test_questions_per_batch_is_editable(self):
        self.assertEqual(self.apply("question_generation.questions_per_batch", "4"), 4)
        self.assertEqual(self.config.question_generation.questions_per_batch, 4)

    def test_pairs_and_lists(self):
        self.assertEqual(self.apply("optimizer.top_k_bounds", "3 12"), (3, 12))
        self.assertEqual(self.apply("optimizer.top_k_bounds", "3, 12"), (3, 12))
        self.assertEqual(
            self.apply("optimizer.similarity_threshold_bounds", "0.2, 0.8"), (0.2, 0.8)
        )
        self.assertEqual(
            self.apply("question_generation.question_types", "factual, inferential"),
            ("factual", "inferential"),
        )

    def test_non_numeric_input_is_rejected(self):
        with self.assertRaises(operations.ConfigEditError):
            self.apply("optimizer.max_iterations", "twelve")
        with self.assertRaises(operations.ConfigEditError):
            self.apply("retrieval.similarity_threshold", "high")
        with self.assertRaises(operations.ConfigEditError):
            self.apply("optimizer.seed_from_profile", "maybe")

    def test_a_pair_needs_exactly_two_ascending_values(self):
        for bad in ("5", "1 2 3", "10 2"):
            with self.subTest(value=bad), self.assertRaises(operations.ConfigEditError):
                self.apply("optimizer.top_k_bounds", bad)

    def test_an_unknown_choice_is_rejected(self):
        with self.assertRaises(operations.ConfigEditError):
            self.apply("retrieval.retriever_type", "magic")
        self.assertEqual(self.apply("retrieval.retriever_type", "mmr"), "mmr")

    def test_an_unknown_field_is_rejected(self):
        with self.assertRaises(operations.ConfigEditError):
            self.apply("retrieval.nonexistent_knob", "1")


class ValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = AppConfig()
        self.config.retrieval.chunk_size = 800
        self.config.retrieval.chunk_overlap = 100

    def test_overlap_must_stay_below_chunk_size(self):
        with self.assertRaises(operations.ConfigEditError) as caught:
            operations.apply_field(self.config, "retrieval.chunk_overlap", "800")
        self.assertIn("chunk_size", str(caught.exception))
        # Rejected means unchanged -- a later save must not persist half an edit.
        self.assertEqual(self.config.retrieval.chunk_overlap, 100)

    def test_a_batch_of_edits_is_validated_as_a_set(self):
        # Raising the overlap past the OLD chunk_size is legitimate when the
        # chunk_size is being raised in the same submission. Validating edit by
        # edit would reject this.
        operations.apply_fields(
            self.config,
            {"retrieval.chunk_size": "1600", "retrieval.chunk_overlap": "400"},
        )
        self.assertEqual(self.config.retrieval.chunk_size, 1600)
        self.assertEqual(self.config.retrieval.chunk_overlap, 400)

    def test_a_rejected_batch_commits_nothing(self):
        with self.assertRaises(operations.ConfigEditError):
            operations.apply_fields(
                self.config,
                {"retrieval.top_k": "5", "retrieval.chunk_overlap": "9000"},
            )
        self.assertEqual(self.config.retrieval.top_k, AppConfig().retrieval.top_k)
        self.assertEqual(self.config.retrieval.chunk_overlap, 100)

    def test_non_positive_numbers_are_rejected(self):
        for path, value in (
            ("retrieval.chunk_size", "0"),
            ("retrieval.top_k", "0"),
            ("optimizer.max_iterations", "0"),
            ("question_generation.questions_per_batch", "0"),
            ("question_generation.batch_size_chars", "-1"),
            ("mistral.requests_per_minute", "0"),
        ):
            with self.subTest(path=path), self.assertRaises(operations.ConfigEditError):
                operations.apply_field(self.config, path, value)

    def test_similarity_threshold_stays_in_range(self):
        with self.assertRaises(operations.ConfigEditError):
            operations.apply_field(self.config, "retrieval.similarity_threshold", "1.5")
        operations.apply_field(self.config, "retrieval.similarity_threshold", "0.5")

    def test_a_prompt_template_missing_a_placeholder_is_rejected(self):
        for bad in (
            "Answer using {context}.",
            "Answer the question {question}.",
            "Use {context} and {context} to answer {question}.",
        ):
            with self.subTest(template=bad), self.assertRaises(operations.ConfigEditError):
                operations.apply_field(self.config, "retrieval.prompt_template", bad)

    def test_a_valid_prompt_template_is_accepted(self):
        good = "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        operations.apply_field(self.config, "retrieval.prompt_template", good)
        self.assertEqual(self.config.retrieval.prompt_template, good)

    def test_the_default_config_is_valid(self):
        # A default that its own editor would reject would make every first edit
        # fail on an unrelated pre-existing problem.
        self.assertEqual(operations.validate(AppConfig()), [])


class DisplayTests(unittest.TestCase):
    def test_long_text_is_truncated_for_a_one_line_listing(self):
        config = AppConfig()
        fields = {f.path: f for f in operations.editable_fields(config)}
        shown = fields["retrieval.prompt_template"].display_value
        self.assertNotIn("\n", shown)
        self.assertLessEqual(len(shown), 60)

    def test_pairs_render_as_comma_separated_values(self):
        config = AppConfig()
        fields = {f.path: f for f in operations.editable_fields(config)}
        self.assertEqual(fields["optimizer.top_k_bounds"].display_value, "2, 10")


class ListLogHandlerTests(unittest.TestCase):
    def test_records_are_captured_and_capped(self):
        import logging

        handler = operations.ListLogHandler(limit=5)
        logger = logging.getLogger("phoenix_rag.test_operations")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        try:
            for index in range(20):
                logger.info("message %d", index)
        finally:
            logger.removeHandler(handler)

        self.assertEqual(len(handler.records), 5)
        self.assertIn("message 19", handler.text())
        self.assertNotIn("message 0 ", handler.text())


if __name__ == "__main__":
    unittest.main()
