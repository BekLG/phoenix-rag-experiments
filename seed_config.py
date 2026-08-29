"""
seed_config.py
==============
Derives the STARTING retrieval configuration from the document profile, so
iteration 1 begins in the regime the document actually calls for instead of at
a document-agnostic default.

WHY THIS EXISTS
---------------
Before this module, every run started from the retrieval block in
config/default_config.json -- chunk_size=300, chunk_overlap=50, top_k=1 -- no
matter what document was loaded. The LLM optimizer then anchored on that value.
On a 12-page paper (doc_type=paper, median_chars_per_page=4353,
chars_per_section~3731), where llm_optimizer's DOCUMENT_CONDITIONED_SIZING_POLICY
explicitly calls for chunk_size 800-1200, ten consecutive iterations never left
the 300-500 band: 300, 500, 400, 350, 300, 400, 350, 500, 450, 300. The whole
optimization budget was spent nudging around the starting value -- which was
also the chunk_size_bounds floor, so downward exploration was impossible and 300
was structurally "home base".

Stating the sizing rules in the prompt was not enough to break that anchor.
Applying them to iteration 1 directly is.

RELATIONSHIP TO THE SIZING POLICY
---------------------------------
The rules below are the executable form of
llm_optimizer.DOCUMENT_CONDITIONED_SIZING_POLICY rules 1-4. They deliberately
stay deterministic: no LLM call, so seeding costs nothing, is reproducible
across runs, and is unit-testable. Both this module and the policy's prompt
context read their profile arithmetic from document_profile.chars_per_section /
estimate_chunk_count, so the numbers the LLM is shown and the numbers the seed
was derived from cannot drift apart.

WHAT IS NOT SEEDED
------------------
retriever_type, similarity_threshold and prompt_template pass through from the
base config untouched. Choosing between similarity / mmr / threshold retrieval
depends on measured precision-vs-recall behaviour, and writing a document-scoped
prompt depends on observed faithfulness failures -- neither of which exists at
iteration 1. The optimizer picks those up from iteration 2 onward.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from config import OptimizerConfig, RetrievalConfig
from document_profile import DocumentProfile, chars_per_section, estimate_chunk_count
from optimizer import _clamp

logger = logging.getLogger("phoenix_rag.seed_config")

PROSE_DOC_TYPES = frozenset({"paper", "narrative", "policy"})

# Regime -> (fraction of chars_per_section, chunk_size band, overlap fraction).
# Bands and fractions come from DOCUMENT_CONDITIONED_SIZING_POLICY rules 1-3:
# a chunk should hold one complete unit of reasoning (a quarter to a half of a
# section), and overlap should carry roughly one sentence of lead-in across the
# boundary -- proportionally less for larger chunks.
_REGIMES = {
    "prose-heavy": (0.33, (800, 1200), 0.15),
    "mixed": (0.33, (600, 900), 0.18),
    "fact-dense": (0.375, (300, 600), 0.20),
}

# Retrieve roughly this share of the document per question. top_k is then
# clamped to the configured bounds and capped against the chunk count.
_TARGET_COVERAGE = 0.10
_MIN_SEED_TOP_K = 3


def _round_to(value: float, step: int) -> int:
    """Round to a human-legible multiple, never below one step."""
    return max(step, int(round(value / step)) * step)


@dataclass(frozen=True)
class SeedProposal:
    """A starting configuration plus why it was chosen.

    `rationale` is written to satisfy the same requirement the sizing policy
    puts on the LLM's own proposals (rule 5): name the profile numbers used and
    the regime concluded. It is logged, and folded into iteration 1's
    applied_rules so later iterations can see what the starting point was
    reasoning from rather than treating it as an arbitrary default.
    """

    config: RetrievalConfig
    regime: str
    rationale: str


def _select_regime(profile: DocumentProfile, section_chars: int) -> str:
    """Classify the document per sizing-policy rule 2.

    Both signal families can fire at once (a policy document full of numbered
    lists, say). The policy tells the LLM to say so and pick; deterministically
    we take the middle band and name "mixed" in the rationale, which is honest
    about the disagreement instead of silently favouring one signal.
    """
    median = profile.median_chars_per_page or 0.0

    prose_signal = profile.doc_type in PROSE_DOC_TYPES or median > 3000
    lookup_signal = (
        profile.doc_type == "manual"
        or profile.table_heavy
        or profile.list_heavy
        or (profile.doc_type == "unknown" and median < 2500)
    )

    if prose_signal and lookup_signal:
        return "mixed"
    if prose_signal:
        return "prose-heavy"
    if lookup_signal:
        return "fact-dense"
    # Neither fired -- e.g. doc_type=unknown with a 2500-3000 median, which
    # falls in the gap between the policy's two thresholds. Section length is
    # the remaining evidence about how long a unit of meaning actually is.
    return "prose-heavy" if section_chars >= 2000 else "fact-dense"


def _seed_top_k(
    characters: int, chunk_size: int, chunk_overlap: int, bounds: tuple[int, int]
) -> int:
    """Pick top_k from document size, then apply sizing-policy rule 4's guard.

    Rule 4 warns that a very low chunk count means top_k pulls a large fraction
    of the whole document and precision suffers. So on top of the coverage
    target, top_k is capped at a quarter of the available chunks -- without that
    cap a two-chunk document would retrieve both and "retrieval" would be
    meaningless.
    """
    lower, _upper = bounds
    if characters <= 0:
        return int(_clamp(lower, bounds))

    by_coverage = max(
        _MIN_SEED_TOP_K, int(round(_TARGET_COVERAGE * characters / chunk_size))
    )

    chunk_count = estimate_chunk_count(characters, chunk_size, chunk_overlap)
    if chunk_count:
        by_coverage = min(by_coverage, max(lower, chunk_count // 4))

    return int(_clamp(by_coverage, bounds))


def propose_seed_config(
    base_config: RetrievalConfig,
    profile: DocumentProfile,
    opt_config: OptimizerConfig,
) -> SeedProposal:
    """Derive iteration 1's chunk_size, chunk_overlap and top_k from `profile`.

    `base_config` supplies every field that is not profile-derivable
    (retriever_type, similarity_threshold, prompt_template) and is never
    mutated -- a copy is returned. Every numeric result is clamped to
    `opt_config`'s bounds, so a seed can never start the run outside the space
    the optimizer is allowed to search.
    """
    section_chars = chars_per_section(profile)
    regime = _select_regime(profile, section_chars)
    fraction, (band_low, band_high), overlap_fraction = _REGIMES[regime]

    chunk_size = _round_to(section_chars * fraction, 50)
    chunk_size = int(_clamp(chunk_size, (band_low, band_high)))
    chunk_size = int(_clamp(chunk_size, opt_config.chunk_size_bounds))

    # Overlap must stay strictly below chunk_size -- chunking.py would otherwise
    # never advance through the document.
    chunk_overlap = min(_round_to(chunk_size * overlap_fraction, 10), chunk_size - 1)
    chunk_overlap = max(0, chunk_overlap)

    top_k = _seed_top_k(
        profile.characters, chunk_size, chunk_overlap, opt_config.top_k_bounds
    )

    seeded = base_config.copy_with(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
    )

    rationale = (
        f"seed_from_profile[{regime}]: doc_type={profile.doc_type}, "
        f"median_chars_per_page={profile.median_chars_per_page}, "
        f"characters={profile.characters}, sections={profile.sections}, "
        f"chars_per_section~{section_chars} -> {regime}, so chunk_size={chunk_size}, "
        f"chunk_overlap={chunk_overlap} ({overlap_fraction:.0%} of chunk_size), "
        f"top_k={top_k} (~{_TARGET_COVERAGE:.0%} document coverage over "
        f"{estimate_chunk_count(profile.characters, chunk_size, chunk_overlap)} "
        f"estimated chunks). retriever_type/similarity_threshold/prompt_template "
        f"left at base values -- those need measured scores to choose."
    )

    logger.info("Profile-seeded starting configuration: %s", rationale)
    return SeedProposal(config=seeded, regime=regime, rationale=rationale)
