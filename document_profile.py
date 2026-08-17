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
