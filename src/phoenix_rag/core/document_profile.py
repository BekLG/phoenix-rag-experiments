"""Compute and cache deterministic document-level retrieval characteristics."""

from __future__ import annotations

import json
import logging
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger("phoenix_rag.document_profile")


@dataclass(frozen=True)
class DocumentProfile:
    pages: int | None
    characters: int
    estimated_tokens: int
    sections: int | None
    median_chars_per_page: float | None
    min_chars_per_page: int | None
    max_chars_per_page: int | None
    doc_type: str
    table_heavy: bool
    list_heavy: bool


_MARKDOWN_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+\S+")
_NUMBERED_HEADING = re.compile(
    r"^\s*\d+(?:\.\d+)*(?:[.)])?\s+[A-Z][^\n]{0,100}$"
)
_LIST_MARKER = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")


def _detect_sections(lines: list[str]) -> int | None:
    headings = sum(
        bool(_MARKDOWN_HEADING.match(line) or _NUMBERED_HEADING.match(line))
        for line in lines
    )
    return headings or None


def _classify_document(full_text: str) -> str:
    """Classify using deterministic keyword signals; no LLM call is made."""
    text = full_text.lower()
    signals = {
        "paper": ("abstract", "methodology", "references", "doi", "et al."),
        "manual": ("installation", "troubleshooting", "user guide", "step 1", "chapter"),
        "policy": ("policy", "shall", "compliance", "effective date", "scope"),
        "narrative": ("chapter", "once upon", "narrator", "dialogue", "protagonist"),
    }
    scores = {name: sum(term in text for term in terms) for name, terms in signals.items()}
    best_type = max(scores, key=scores.get)
    return best_type if scores[best_type] >= 2 else "unknown"


def _page_lengths(full_text: str, pages: int | None) -> list[int] | None:
    if pages is None or pages <= 0:
        return None
    separated = full_text.split("\f")
    if len(separated) == pages:
        return [len(page) for page in separated]

    # Joined loader output does not retain page boundaries. Distribute the
    # known total deterministically so these fields remain useful estimates.
    base, remainder = divmod(len(full_text), pages)
    return [base + (1 if index < remainder else 0) for index in range(pages)]


def compute_profile(full_text: str, pages: int | None = None) -> DocumentProfile:
    lines = [line for line in full_text.splitlines() if line.strip()]
    table_lines = sum(line.count("|") >= 2 or "\t" in line for line in lines)
    list_lines = sum(bool(_LIST_MARKER.match(line)) for line in lines)
    line_count = len(lines)
    lengths = _page_lengths(full_text, pages)

    return DocumentProfile(
        pages=pages if pages is None or pages > 0 else None,
        characters=len(full_text),
        # Four characters per token is a deliberately simple planning heuristic.
        estimated_tokens=len(full_text) // 4,
        sections=_detect_sections(lines),
        median_chars_per_page=float(statistics.median(lengths)) if lengths else None,
        min_chars_per_page=min(lengths) if lengths else None,
        max_chars_per_page=max(lengths) if lengths else None,
        doc_type=_classify_document(full_text),
        table_heavy=line_count > 0 and table_lines >= 2 and table_lines / line_count >= 0.15,
        list_heavy=line_count > 0 and list_lines >= 2 and list_lines / line_count >= 0.20,
    )


def chars_per_section(profile: DocumentProfile) -> int:
    """The document's natural unit of meaning: characters per detected section.

    Falls back to the whole document when no headings were detected, so callers
    never have to guard against `sections` being None or 0.
    """
    return profile.characters // profile.sections if profile.sections else profile.characters


def estimate_chunk_count(characters: int, chunk_size: int, chunk_overlap: int) -> int:
    """How many chunks a chunk_size/overlap pair yields over `characters`.

    An estimate, not a count: the real splitter honours separators, so it lands
    near this rather than on it.
    """
    if characters <= 0:
        return 0
    effective_step = max(1, chunk_size - chunk_overlap)
    return (characters + effective_step - 1) // effective_step


def aggregate_profiles(profiles: list[DocumentProfile]) -> DocumentProfile:
    """Collapse per-document profiles into one profile describing the corpus.

    Used when retrieval spans several documents (corpus.py): the optimizer and
    seed_config both take a single DocumentProfile, and the thing they need to
    size chunks for is the whole searchable corpus, not any one member of it.

    Deterministic, no LLM call -- same contract as compute_profile.

    Two choices worth stating outright:

    * `doc_type` is the character-weighted dominant type, and is NEVER a new
      label like "mixed". seed_config._select_regime and llm_optimizer's sizing
      policy both branch on the exact vocabulary compute_profile produces, so
      inventing a value here would silently drop the corpus into their fallback
      paths. The actual composition is reported in the corpus summary text
      instead, where the LLM can read it.
    * `median_chars_per_page` is the page-weighted median of the per-document
      medians. That is exact under the same uniform-page-length assumption
      _page_lengths already makes whenever a loader hands us joined text with no
      form-feeds, which is the common case.
    """
    if not profiles:
        return compute_profile("")
    if len(profiles) == 1:
        return profiles[0]

    def _sum_or_none(values: list[int | None]) -> int | None:
        present = [v for v in values if v is not None]
        return sum(present) if present else None

    characters = sum(p.characters for p in profiles)

    # Page-weighted median: repeat each document's median once per page it has,
    # falling back to one vote per document when page counts are unknown.
    weighted_medians: list[float] = []
    for profile in profiles:
        if profile.median_chars_per_page is None:
            continue
        weight = profile.pages if profile.pages and profile.pages > 0 else 1
        weighted_medians.extend([profile.median_chars_per_page] * weight)

    per_page_mins = [p.min_chars_per_page for p in profiles if p.min_chars_per_page is not None]
    per_page_maxes = [p.max_chars_per_page for p in profiles if p.max_chars_per_page is not None]

    # Character-weighted vote, so a 50-page paper is not outvoted by a one-page
    # note. Ties break on the first-registered document, which keeps the result
    # stable across runs for a fixed manifest order.
    type_weights: dict[str, int] = {}
    for profile in profiles:
        type_weights[profile.doc_type] = type_weights.get(profile.doc_type, 0) + profile.characters
    dominant_type = max(type_weights, key=type_weights.get) if type_weights else "unknown"

    def _weighted_flag(attribute: str) -> bool:
        if characters <= 0:
            return False
        heavy = sum(p.characters for p in profiles if getattr(p, attribute))
        return heavy / characters >= 0.5

    return DocumentProfile(
        pages=_sum_or_none([p.pages for p in profiles]),
        characters=characters,
        estimated_tokens=sum(p.estimated_tokens for p in profiles),
        sections=_sum_or_none([p.sections for p in profiles]),
        median_chars_per_page=(
            float(statistics.median(weighted_medians)) if weighted_medians else None
        ),
        min_chars_per_page=min(per_page_mins) if per_page_mins else None,
        max_chars_per_page=max(per_page_maxes) if per_page_maxes else None,
        doc_type=dominant_type,
        table_heavy=_weighted_flag("table_heavy"),
        list_heavy=_weighted_flag("list_heavy"),
    )


def save_profile(profile: DocumentProfile, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(profile), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Saved document profile to %s", path)


def load_profile(path: str | Path) -> DocumentProfile:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return DocumentProfile(**data)


def get_or_create_profile(
    full_text: str,
    profile_path: str | Path,
    pages: int | None = None,
    force_regenerate: bool = False,
) -> DocumentProfile:
    path = Path(profile_path)
    if path.exists() and path.stat().st_size > 0 and not force_regenerate:
        logger.info("Loading cached document profile from %s", path)
        return load_profile(path)

    if path.exists() and path.stat().st_size == 0:
        logger.warning("Cached document profile at %s is empty; recomputing", path)

    profile = compute_profile(full_text, pages=pages)
    save_profile(profile, path)
    return profile
