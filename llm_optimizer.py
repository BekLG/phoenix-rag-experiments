"""
llm_optimizer.py
================
LLM-driven optimizer for Phoenix RAG's retrieval configuration. This is now
the ONLY optimizer used by experiment_runner.py -- the rule-based
optimizer (optimizer.propose_next_config) and its bolted-on prompt-refiner
escape hatch (prompt_refiner.refine_prompt) are no longer invoked anywhere
in the loop.

The LLM is shown the FULL iteration history -- every past config, every
resulting score, and what changed each time -- and asked to propose the
next configuration directly. That gives it two things the rule-based
version structurally couldn't have:
  1. Visibility across all four metrics simultaneously, not just whichever
     one has the biggest gap this round.
  2. Memory of trade-offs already observed (e.g. "switching to
     similarity_score_threshold raised precision but tanked recall last
     time") so it can avoid repeating a fix that already backfired.

Prompt tuning is no longer a boolean toggle between "default" and
COMBINED_PROMPT. The LLM is given the document summary, a static
DocumentProfile, the current RetrievalConfig, and the full iteration history.
It is asked to WRITE the prompt template itself -- the same way it writes
chunk_size or top_k -- so the prompt can be tailored to the actual document
and to whichever generation-quality metric is struggling.

There is no rule-based fallback anymore. If the LLM's response can't be
parsed, fails validation, or proposes a prompt template missing/duplicating
the required {context}/{question} placeholders, this module simply keeps
the current configuration unchanged for that iteration (with a rule string
explaining why) rather than silently handing control to a different
optimization strategy.
"""

from __future__ import annotations

import json
import logging
import re

from config import GeminiSettings, OptimizerConfig, RetrievalConfig
from document_profile import DocumentProfile
from gemini_client import GeminiClient
from optimizer import _clamp, meets_targets

logger = logging.getLogger("phoenix_rag.llm_optimizer")

REQUIRED_PLACEHOLDERS = ("{context}", "{question}")

SYSTEM_PROMPT = """You are an expert RAG (Retrieval-Augmented Generation) \
systems engineer, tuning both the retrieval configuration AND the answer \
prompt template of a RAG pipeline to maximize four Ragas evaluation \
metrics, each with a target:

  - faithfulness: does the generated answer stay grounded in the retrieved \
context, without inventing facts? Raised by a stricter/more constrained \
prompt template, or by retrieving less noisy/irrelevant context (lower \
top_k).
  - context_recall: did retrieval find the information actually needed to \
answer the question? Raised by retrieving more chunks (top_k), or by using \
smaller chunks (chunk_size) so more distinct pieces of information can be \
retrieved within the same top_k budget.
  - context_precision: how much of what was retrieved was actually useful, \
vs noise? Raised by retrieving fewer chunks (top_k), or by switching to a \
similarity_score_threshold retriever that filters out low-relevance chunks.
  - response_relevancy: does the generated answer directly address the \
question, without padding or drifting off-topic? Raised by a stricter, \
more focused prompt template, or by reducing top_k to cut noisy context.

IMPORTANT -- you are given the FULL history of past iterations below. Use it:
  - Increasing top_k tends to help recall but can hurt precision.
  - Switching to similarity_score_threshold retrieval tends to help \
precision but can hurt recall if the threshold excludes relevant chunks.
  - If a past change made a metric worse, do not blindly repeat that same \
direction. If two metrics are visibly trading off against each other \
across iterations, propose a configuration that balances both rather than \
repeatedly over-correcting one at the other's expense.
  - If the history shows the same configuration being re-tried with only \
noise-level score differences, propose something genuinely different \
rather than a small nudge.
  - If faithfulness or response_relevancy is the weak metric, do not just \
flip between two fixed prompts -- WRITE a new prompt template, tailored to \
the document summary you're given and to the specific failure mode, that \
more tightly constrains the model (e.g. explicit grounding instructions, \
narrower scope, an explicit "say you don't know" instruction) or refocuses \
it on the question. Reuse a past prompt only if nothing about the failure \
mode has changed since it was tried.

You will be given the tunable parameter bounds, a summary and static profile \
of the source document, the current configuration and its live estimated \
chunk count, and the full iteration history. Propose the next configuration \
to try.

Respond with ONLY a JSON object (no markdown fences, no commentary) with \
exactly these keys:
  "chunk_size": integer within bounds
  "chunk_overlap": integer, must be less than chunk_size
  "top_k": integer within bounds
  "retriever_type": one of "similarity", "mmr", "similarity_score_threshold"
  "similarity_threshold": float within bounds (only meaningful if \
retriever_type is "similarity_score_threshold", but always include a value)
  "prompt_template": the FULL answer prompt template to use next, as a \
single string. It MUST contain exactly one occurrence each of the literal \
placeholders {context} and {question} -- these get filled in \
programmatically at answer time, do not remove, rename, or duplicate them. \
You may reuse the current prompt template unchanged if you don't believe \
it needs to change this iteration.
  "reasoning": a short (1-2 sentence) explanation of why you chose these \
values given the history
"""


def _format_history(history: list[dict]) -> str:
    if not history:
        return "(no iterations run yet -- this is the first configuration)"

    lines = []
    for row in history:
        cfg = row["config"]
        scores = row["scores"]
        lines.append(
            f"Iteration {row['iteration']}: "
            f"chunk_size={cfg['chunk_size']}, chunk_overlap={cfg['chunk_overlap']}, "
            f"top_k={cfg['top_k']}, similarity_threshold={cfg['similarity_threshold']}, "
            f"retriever_type={cfg['retriever_type']}, "
            f"prompt_template={cfg['prompt_template']!r} "
            f"-> faithfulness={scores.get('faithfulness', 0):.3f}, "
            f"context_recall={scores.get('context_recall', 0):.3f}, "
            f"context_precision={scores.get('context_precision', 0):.3f}, "
            f"response_relevancy={scores.get('response_relevancy', 0):.3f} "
            f"[{row.get('applied_rules', '')}]"
        )
    return "\n".join(lines)


def _format_bounds(opt_config: OptimizerConfig) -> str:
    return (
        f"chunk_size bounds: {opt_config.chunk_size_bounds}\n"
        f"top_k bounds: {opt_config.top_k_bounds}\n"
        f"similarity_threshold bounds: {opt_config.similarity_threshold_bounds}\n"
        f"Targets -- faithfulness >= {opt_config.target_faithfulness}, "
        f"context_recall >= {opt_config.target_context_recall}, "
        f"context_precision >= {opt_config.target_context_precision}, "
        f"response_relevancy >= {opt_config.target_response_relevancy}"
    )


def _format_profile(profile: DocumentProfile) -> str:
    """Render static document facts as compact structured prompt context."""
    fields = (
        f"pages={profile.pages}",
        f"characters={profile.characters}",
        f"estimated_tokens={profile.estimated_tokens}",
        f"sections={profile.sections}",
        f"median_chars_per_page={profile.median_chars_per_page}",
        f"min_chars_per_page={profile.min_chars_per_page}",
        f"max_chars_per_page={profile.max_chars_per_page}",
        f"doc_type={profile.doc_type}",
        f"table_heavy={profile.table_heavy}",
        f"list_heavy={profile.list_heavy}",
    )
    return "{" + ", ".join(fields) + "}"


def _estimate_chunk_count(
    characters: int, chunk_size: int, chunk_overlap: int
) -> int:
    effective_step = max(1, chunk_size - chunk_overlap)
    if characters <= 0:
        return 0
    return (characters + effective_step - 1) // effective_step


def _format_dynamic_context(
    current_config: RetrievalConfig, profile: DocumentProfile
) -> str:
    estimated_count = _estimate_chunk_count(
        profile.characters,
        current_config.chunk_size,
        current_config.chunk_overlap,
    )
    return (
        "CURRENT RETRIEVAL SHAPE:\n"
        f"estimated_chunk_count={estimated_count} "
        "(computed from document characters and current chunk settings)"
    )


def _validate_prompt_template(template: str) -> bool:
    """A proposed prompt template is only usable if it kept both required
    placeholders, each exactly once. Silently accepting a broken template
    (missing or duplicated placeholders) would crash str.format() downstream
    in rag_pipeline.py's build_prompt(), so this is checked before the
    template is ever installed into a RetrievalConfig.
    """
    return isinstance(template, str) and all(
        template.count(ph) == 1 for ph in REQUIRED_PLACEHOLDERS
    )


def _parse_llm_config(raw: str) -> dict | None:
    """Best-effort JSON parse, tolerant of stray markdown fences.

    Same defensive pattern as question_generator.py's _parse_llm_json.
    """
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return data if isinstance(data, dict) else None


def propose_next_config_llm(
    current_config: RetrievalConfig,
    scores: dict[str, float],
    opt_config: OptimizerConfig,
    gemini_settings: GeminiSettings,
    history: list[dict],
    document_summary: str,
    document_profile: DocumentProfile,
) -> tuple[RetrievalConfig, list[str]]:
    """LLM-driven proposer for the next configuration -- the sole optimizer
    used by experiment_runner.py.

    `history` is a list of dicts, one per past iteration, each shaped:
        {"iteration": int, "config": RetrievalConfig.to_dict(),
         "scores": dict, "applied_rules": str}
    experiment_runner.py builds this incrementally as the run progresses.

    `document_summary` lets the LLM tailor the prompt_template it writes to
    what the source document actually is, the same way prompt_refiner.py
    used to, except now it's just one more field the LLM proposes on every
    iteration instead of a separate escape-hatch call gated behind two
    other tiers being exhausted.

    `document_profile` provides static, deterministic retrieval-shape facts;
    the current configuration's estimated chunk count is computed live from
    that profile and is never stored in it.

    There is no rule-based fallback. If the LLM's response can't be parsed,
    fails validation (missing keys, invalid retriever_type, non-numeric
    values, a malformed prompt_template), or is otherwise unusable, this
    returns the CURRENT configuration unchanged along with a rule string
    explaining why, so a bad LLM response never crashes a run or gets
    silently applied -- and every numeric value that IS present is also
    re-clamped to its configured bounds even on an otherwise well-formed
    response.
    """
    if meets_targets(scores, opt_config):
        logger.info("All metrics meet target thresholds.")
        return current_config, ["all_targets_met"]

    client = GeminiClient(gemini_settings)

    user_message = (
        f"{_format_bounds(opt_config)}\n\n"
        f"DOCUMENT SUMMARY:\n{document_summary}\n\n"
        f"DOCUMENT PROFILE:\n{_format_profile(document_profile)}\n\n"
        f"{_format_dynamic_context(current_config, document_profile)}\n\n"
        f"Current configuration: {current_config.to_dict()}\n\n"
        f"Iteration history:\n{_format_history(history)}\n\n"
        "Propose the next configuration."
    )

    raw = client.chat(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        model=gemini_settings.optimizer_model,
        temperature=0.4,
    )

    parsed = _parse_llm_config(raw)
    if parsed is None:
        logger.warning(
            "LLM optimizer response unparseable, keeping current configuration."
        )
        return current_config, ["llm_optimizer: response unparseable, kept current config"]

    try:
        chunk_size = int(_clamp(int(parsed["chunk_size"]), opt_config.chunk_size_bounds))
        top_k = int(_clamp(int(parsed["top_k"]), opt_config.top_k_bounds))
        similarity_threshold = float(
            _clamp(float(parsed["similarity_threshold"]), opt_config.similarity_threshold_bounds)
        )

        retriever_type = parsed["retriever_type"]
        if retriever_type not in ("similarity", "mmr", "similarity_score_threshold"):
            raise ValueError(f"invalid retriever_type: {retriever_type!r}")

        chunk_overlap = int(parsed.get("chunk_overlap", current_config.chunk_overlap))
        chunk_overlap = max(0, min(chunk_overlap, chunk_size - 1))

    except (KeyError, TypeError, ValueError) as exc:
        logger.warning(
            "LLM optimizer response failed validation (%s), keeping current configuration.",
            exc,
        )
        return current_config, [f"llm_optimizer: validation failed ({exc}), kept current config"]

    proposed_prompt = parsed.get("prompt_template")
    if _validate_prompt_template(proposed_prompt):
        prompt_template = proposed_prompt
        prompt_note = "prompt_template_updated"
    else:
        prompt_template = current_config.prompt_template
        prompt_note = "prompt_template_invalid_kept_current"
        logger.warning(
            "LLM-proposed prompt_template missing/duplicating required placeholders %s; "
            "keeping existing prompt template.",
            REQUIRED_PLACEHOLDERS,
        )

    next_config = current_config.copy_with(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
        retriever_type=retriever_type,
        prompt_template=prompt_template,
    )

    reasoning = str(parsed.get("reasoning", "")).strip()
    rule_desc = (
        f"llm_optimizer[{prompt_note}]: {reasoning}"
        if reasoning
        else f"llm_optimizer[{prompt_note}]: (no reasoning given)"
    )

    logger.info("LLM optimizer proposed: %s | %s", next_config.to_dict(), rule_desc)
    return next_config, [rule_desc]
