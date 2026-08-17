"""
question_generator.py
======================
Generates the fixed evaluation benchmark used throughout optimization.

CRITICAL DESIGN RULE (do not violate):
    Questions are generated from the COMPLETE source document (split into
    batches purely for LLM context-window / rate-limit reasons), NEVER
    from vector search results. If questions were generated from
    retrieved chunks, the benchmark would be biased toward whatever the
    current retrieval configuration happens to surface, making it
    impossible to fairly compare configurations against each other.

Workflow:
    1. Split the full document text into batches (character-based, not
       the same as the retrieval chunker).
    2. For each batch, prompt Mistral for a mix of question types.
    3. Merge all batches, deduplicate near-identical questions.
    4. Persist to disk. Later runs load this file instead of regenerating
       it (unless `regenerate_each_iteration` / `force` is set).

Generation calls go through MistralClient (mistral_client.py) rather than
a raw SDK client, so rate limiting and retry/backoff actually apply here.
Without this, a 429 mid-batch is caught by the broad except-Exception below
and that batch's questions are silently lost rather than retried.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from config import MistralSettings, QuestionGenerationConfig
from mistral_client import MistralClient
from storage import _atomic_write_json

logger = logging.getLogger("phoenix_rag.question_generator")


@dataclass
class BenchmarkQuestion:
    question: str
    reference_answer: str
    reference_context: str
    question_type: str


SYSTEM_PROMPT = """You are an expert evaluation-question writer for a RAG \
(Retrieval-Augmented Generation) benchmark. You will be given a chunk of a \
source document. Generate diverse, high-quality questions that can be \
answered using ONLY the given text.

Return ONLY a JSON array (no markdown fences, no commentary) where each \
element has exactly these keys:
  - "question": the question text
  - "reference_answer": a correct, concise answer grounded in the text
  - "reference_context": the exact sentence(s) from the text that support the answer
  - "question_type": one of {question_types}

Aim for a mix of question types across the array. Do not invent facts not \
present in the text."""


def _batch_text(full_text: str, batch_size_chars: int) -> list[str]:
    """Split the full document text into character-based batches.

    This is intentionally independent from the retrieval chunker
    (chunking.py) -- it exists purely to keep each LLM call within a
    reasonable context size / rate limit, not to define retrieval units.
    """
    batches = []
    for start in range(0, len(full_text), batch_size_chars):
        batch = full_text[start : start + batch_size_chars].strip()
        if batch:
            batches.append(batch)
    return batches


def _parse_llm_json(raw: str) -> list[dict]:
    """Best-effort parse of the LLM's JSON array, tolerant of stray fences."""
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to salvage the first [...] block in the response
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if not match:
            logger.warning("Could not parse question batch response as JSON")
            return []
        data = json.loads(match.group(0))
    return data if isinstance(data, list) else []


def _generate_for_batch(
    client: MistralClient,
    batch_text: str,
    qg_config: QuestionGenerationConfig,
    mistral_settings: MistralSettings,
) -> list[BenchmarkQuestion]:
    system = SYSTEM_PROMPT.format(question_types=", ".join(qg_config.question_types))
    user = (
        f"Generate {qg_config.questions_per_batch} questions from this text:\n\n"
        f"{batch_text}"
    )
    # MistralClient.chat() handles rate limiting + exponential-backoff retry
    # internally and returns the answer text directly (not a raw SDK
    # response object needing .choices[0].message.content).
    raw = client.chat(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        model=mistral_settings.generation_model,
        temperature=0.4,
    )
    items = _parse_llm_json(raw)

    questions = []
    for item in items:
        try:
            questions.append(
                BenchmarkQuestion(
                    question=item["question"],
                    reference_answer=item["reference_answer"],
                    reference_context=item["reference_context"],
                    question_type=item.get("question_type", "unknown"),
                )
            )
        except (KeyError, TypeError):
            logger.warning("Skipping malformed question item: %s", item)
    return questions


def _dedupe(questions: list[BenchmarkQuestion]) -> list[BenchmarkQuestion]:
    """Remove exact/near-exact duplicate questions (case/whitespace-insensitive).

    A lightweight normalized-string dedupe. For stricter semantic
    dedup, swap in an embedding-similarity comparison using
    MistralClient.embed and qg_config.dedup_similarity_threshold.
    """
    seen: set[str] = set()
    unique: list[BenchmarkQuestion] = []
    for q in questions:
        key = re.sub(r"\s+", " ", q.question.strip().lower())
        if key not in seen:
            seen.add(key)
            unique.append(q)
    logger.info("Deduplicated %d -> %d questions", len(questions), len(unique))
    return unique


def generate_benchmark(
    full_text: str,
    mistral_settings: MistralSettings,
    qg_config: QuestionGenerationConfig,
) -> list[BenchmarkQuestion]:
    """Generate the full benchmark question set from the complete document."""
    client = MistralClient(mistral_settings)
    batches = _batch_text(full_text, qg_config.batch_size_chars)
    logger.info("Generating questions from %d batch(es)", len(batches))

    all_questions: list[BenchmarkQuestion] = []
    for i, batch in enumerate(batches, start=1):
        logger.info("Generating questions for batch %d/%d", i, len(batches))
        try:
            all_questions.extend(
                _generate_for_batch(client, batch, qg_config, mistral_settings)
            )
        except Exception:
            logger.exception(
                "Batch %d failed to generate questions; continuing with remaining "
                "batches (safe to re-run, already-succeeded batches are not lost "
                "if you persist incrementally)",
                i,
            )

    return _dedupe(all_questions)


def save_benchmark(questions: list[BenchmarkQuestion], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(path, [asdict(q) for q in questions])
    logger.info("Saved %d benchmark questions to %s", len(questions), path)


def load_benchmark(path: str | Path) -> list[BenchmarkQuestion]:
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    return [BenchmarkQuestion(**item) for item in data]


def get_or_create_benchmark(
    full_text: str,
    mistral_settings: MistralSettings,
    qg_config: QuestionGenerationConfig,
    benchmark_path: str | Path,
    force_regenerate: bool = False,
) -> list[BenchmarkQuestion]:
    """Load the cached benchmark unless it doesn't exist or regeneration is forced.

    This is what keeps the benchmark FIXED across optimization iterations:
    every configuration is evaluated against exactly the same questions.
    """
    path = Path(benchmark_path)
    if path.exists() and not force_regenerate and not qg_config.regenerate_each_iteration:
        logger.info("Loading cached benchmark from %s", path)
        return load_benchmark(path)

    questions = generate_benchmark(full_text, mistral_settings, qg_config)
    save_benchmark(questions, path)
    return questions
