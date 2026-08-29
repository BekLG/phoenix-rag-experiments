import unittest

from config import OptimizerConfig, RetrievalConfig
from document_profile import DocumentProfile
from seed_config import propose_seed_config


def _profile(**overrides) -> DocumentProfile:
    """A profile with sane defaults; tests override only what they exercise."""
    fields = {
        "pages": 4,
        "characters": 9599,
        "estimated_tokens": 2399,
        "sections": 12,
        "median_chars_per_page": 2400.0,
        "min_chars_per_page": 2399,
        "max_chars_per_page": 2400,
        "doc_type": "unknown",
        "table_heavy": False,
        "list_heavy": False,
    }
    fields.update(overrides)
    return DocumentProfile(**fields)


# The two real documents this was calibrated against.
DOC_A = _profile()  # "The Abilene Paradox", 4 pages
DOC_B = _profile(
    pages=12,
    characters=52238,
    estimated_tokens=13059,
    sections=14,
    median_chars_per_page=4353.0,
    min_chars_per_page=4353,
    max_chars_per_page=4354,
    doc_type="paper",
)


class RegimeSelectionTests(unittest.TestCase):
    def test_doc_a_is_fact_dense(self):
        """unknown doc_type with a sub-2500 median and short sections."""
        seed = propose_seed_config(RetrievalConfig(), DOC_A, OptimizerConfig())
        self.assertEqual(seed.regime, "fact-dense")
        self.assertEqual(seed.config.chunk_size, 300)
        self.assertEqual(seed.config.chunk_overlap, 60)
        self.assertEqual(seed.config.top_k, 3)

    def test_doc_b_seeds_into_the_prose_band(self):
        """The case that motivated this: 300 was the anchor, 800-1200 was right."""
        seed = propose_seed_config(RetrievalConfig(), DOC_B, OptimizerConfig())
        self.assertEqual(seed.regime, "prose-heavy")
        self.assertEqual(seed.config.chunk_size, 1200)
        self.assertEqual(seed.config.chunk_overlap, 180)
        self.assertEqual(seed.config.top_k, 4)

    def test_long_pages_alone_imply_prose(self):
        profile = _profile(median_chars_per_page=4000.0, characters=40000, sections=10)
        seed = propose_seed_config(RetrievalConfig(), profile, OptimizerConfig())
        self.assertEqual(seed.regime, "prose-heavy")

    def test_manual_with_tables_is_fact_dense(self):
        profile = _profile(
            pages=30,
            characters=90000,
            sections=60,
            median_chars_per_page=3000.0,
            doc_type="manual",
            table_heavy=True,
            list_heavy=True,
        )
        seed = propose_seed_config(RetrievalConfig(), profile, OptimizerConfig())
        self.assertEqual(seed.regime, "fact-dense")
        self.assertEqual(seed.config.chunk_size, 550)
        self.assertEqual(seed.config.chunk_overlap, 110)

    def test_disagreeing_signals_take_the_middle_band(self):
        """doc_type=policy says prose, list_heavy says lookup."""
        profile = _profile(
            pages=8,
            characters=30000,
            sections=20,
            median_chars_per_page=3750.0,
            doc_type="policy",
            list_heavy=True,
        )
        seed = propose_seed_config(RetrievalConfig(), profile, OptimizerConfig())
        self.assertEqual(seed.regime, "mixed")
        self.assertEqual(seed.config.chunk_size, 600)
        self.assertIn("mixed", seed.rationale)

    def test_neither_signal_falls_back_to_section_length(self):
        """unknown doc_type in the 2500-3000 median gap the policy leaves open."""
        long_sections = _profile(median_chars_per_page=2700.0, characters=40000, sections=10)
        short_sections = _profile(median_chars_per_page=2700.0, characters=9000, sections=30)
        self.assertEqual(
            propose_seed_config(RetrievalConfig(), long_sections, OptimizerConfig()).regime,
            "prose-heavy",
        )
        self.assertEqual(
            propose_seed_config(RetrievalConfig(), short_sections, OptimizerConfig()).regime,
            "fact-dense",
        )


class BoundsAndInvariantTests(unittest.TestCase):
    def test_bounds_win_over_the_regime_band(self):
        """A prose seed wants 1200; bounds capped at 500 must take precedence."""
        opt_config = OptimizerConfig(chunk_size_bounds=(300, 500), top_k_bounds=(2, 4))
        seed = propose_seed_config(RetrievalConfig(), DOC_B, opt_config)
        self.assertEqual(seed.config.chunk_size, 500)
        self.assertLessEqual(seed.config.top_k, 4)

    def test_top_k_respects_the_lower_bound(self):
        opt_config = OptimizerConfig(top_k_bounds=(5, 10))
        seed = propose_seed_config(RetrievalConfig(), DOC_A, opt_config)
        self.assertGreaterEqual(seed.config.top_k, 5)

    def test_overlap_always_below_chunk_size(self):
        for profile in (DOC_A, DOC_B, _profile(characters=0, sections=None)):
            seed = propose_seed_config(RetrievalConfig(), profile, OptimizerConfig())
            self.assertLess(seed.config.chunk_overlap, seed.config.chunk_size)
            self.assertGreaterEqual(seed.config.chunk_overlap, 0)

    def test_missing_sections_uses_whole_document(self):
        seed = propose_seed_config(
            RetrievalConfig(), _profile(characters=200, sections=None), OptimizerConfig()
        )
        self.assertEqual(seed.config.chunk_size, 300)
        # One chunk of material: rule 4's guard stops top_k retrieving all of it.
        self.assertEqual(seed.config.top_k, 2)

    def test_empty_document_does_not_divide_by_zero(self):
        seed = propose_seed_config(
            RetrievalConfig(),
            _profile(characters=0, sections=None, median_chars_per_page=0.0),
            OptimizerConfig(),
        )
        self.assertEqual(seed.config.chunk_size, 300)
        self.assertEqual(seed.config.top_k, 2)

    def test_non_derivable_fields_pass_through_untouched(self):
        base = RetrievalConfig(
            retriever_type="mmr",
            similarity_threshold=0.42,
            prompt_template="Custom {context} and {question}",
        )
        seed = propose_seed_config(base, DOC_B, OptimizerConfig())
        self.assertEqual(seed.config.retriever_type, "mmr")
        self.assertEqual(seed.config.similarity_threshold, 0.42)
        self.assertEqual(seed.config.prompt_template, base.prompt_template)
        # The base config itself is never mutated.
        self.assertEqual(base.chunk_size, 300)
        self.assertEqual(base.top_k, 1)

    def test_rationale_cites_the_profile_numbers(self):
        """Sizing-policy rule 5 applied to our own proposals, not just the LLM's."""
        seed = propose_seed_config(RetrievalConfig(), DOC_B, OptimizerConfig())
        for fragment in ("doc_type=paper", "median_chars_per_page=4353.0",
                         "chars_per_section~3731", "prose-heavy", "chunk_size=1200"):
            self.assertIn(fragment, seed.rationale)


if __name__ == "__main__":
    unittest.main()
