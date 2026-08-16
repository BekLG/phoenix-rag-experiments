"""
document_summarizer.py
=======================
Generates and caches a concise summary of the source document.

This exists to give prompt_refiner.py real context about what kind of
document this is -- its subject, structure, and the kind of claims it
makes -- so LLM-driven prompt rewrites are grounded in the actual
document rather than generic advice.

Cached the same way question_generator.py caches the benchmark: generate
once, save to disk, reuse on subsequent runs unless forced.
"""

from __future__ import annotations

import logging
from pathlib import Path

from config import GeminiSettings
from gemini_client import GeminiClient

logger = logging.getLogger("phoenix_rag.document_summarizer")

SUMMARY_SYSTEM_PROMPT = """You are an expert technical summarizer. Given a \
document, produce a concise but information-dense summary (250-400 words) \
covering: the document's main subject, its key sections/topics, the kind \
of factual claims it makes, and its overall structure. This summary will \
be used to help an AI system understand what KIND of document this is, \
not to replace reading it. Do not omit important terminology or names \
that appear in the document."""

# Defensive cap so a very large document doesn't blow the model's context
# window during summarization. Most documents won't hit this.
_MAX_CHARS_FOR_SUMMARY = 40000


def generate_summary(full_text: str, gemini_settings: GeminiSettings) -> str:
    """Generate a concise summary of the full document text."""
    client = GeminiClient(gemini_settings)

    text_for_summary = full_text
    if len(full_text) > _MAX_CHARS_FOR_SUMMARY:
        text_for_summary = (
            full_text[:_MAX_CHARS_FOR_SUMMARY]
            + "\n\n[...document truncated for summarization...]"
        )
        logger.warning(
            "Document is %d chars, truncated to %d for summarization",
            len(full_text), _MAX_CHARS_FOR_SUMMARY,
        )

    summary = client.chat(
        messages=[
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": text_for_summary},
        ],
        model=gemini_settings.generation_model,
        temperature=0.2,
    )
    logger.info("Generated document summary (%d chars)", len(summary))
    return summary


def save_summary(summary: str, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(summary, encoding="utf-8")
    logger.info("Saved document summary to %s", path)


def load_summary(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def get_or_create_summary(
    full_text: str,
    gemini_settings: GeminiSettings,
    summary_path: str | Path,
    force_regenerate: bool = False,
) -> str:
    """Load the cached summary unless it doesn't exist, is empty, or
    regeneration is forced.

    Same fixed-artifact pattern as get_or_create_benchmark() in
    question_generator.py -- generate once, reuse across the whole run
    (and across future runs on the same document). The size check
    specifically guards against a prior crashed write leaving behind an
    empty-but-existing file, which would otherwise be treated as valid
    cached content forever.
    """
    path = Path(summary_path)
    if path.exists() and path.stat().st_size > 0 and not force_regenerate:
        logger.info("Loading cached document summary from %s", path)
        return load_summary(path)

    if path.exists() and path.stat().st_size == 0:
        logger.warning(
            "Cached document summary at %s exists but is empty (likely a "
            "prior crashed write); regenerating.",
            path,
        )

    summary = generate_summary(full_text, gemini_settings)
    save_summary(summary, path)
    return summary
