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

from config import MistralSettings, OptimizerConfig, RetrievalConfig
from document_profile import DocumentProfile, chars_per_section, estimate_chunk_count
from mistral_client import MistralClient
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
top_k). ALSO raised by chunks large enough to hold a complete unit of \
reasoning: a chunk that truncates an argument, definition, or worked \
example mid-thought forces the model to bridge the gap from its own \
knowledge, which scores as unfaithful. If faithfulness is weak while \
context_recall is adequate, suspect chunks that are too SMALL before \
tightening the prompt further.
  - context_recall: did retrieval find the information actually needed to \
answer the question? This is genuinely BIDIRECTIONAL in chunk_size, and \
you must reason about which direction applies:
      * SMALLER chunks pack more distinct facts into the same top_k budget, \
which helps when each answer is a short self-contained span (lookup-style \
questions over fact-dense text).
      * LARGER chunks help when the answer spans several consecutive \
sentences -- comparisons ("what are the two types of X and how do they \
differ"), causal chains, multi-step explanations. Splitting such an answer \
across chunk boundaries loses it even at high top_k, because each fragment \
looks only partly relevant to the retriever.
    Raising top_k also helps recall, but it is the blunter instrument: it \
costs precision, whereas correctly-sized chunks cost nothing.
  - context_precision: how much of what was retrieved was actually useful, \
vs noise? Raised by retrieving fewer chunks (top_k), or by switching to a \
similarity_score_threshold retriever that filters out low-relevance chunks. \
Be aware this metric has a built-in bias toward small chunks -- a large \
chunk carries more material beyond the answer span and can be judged partly \
irrelevant even when it is the RIGHT chunk to retrieve. Do not shrink \
chunk_size purely to chase this metric if faithfulness is falling as a \
result; report the trade-off in your reasoning instead.
  - response_relevancy: does the generated answer directly address the \
question, without padding or drifting off-topic? Raised by a stricter, \
more focused prompt template, or by reducing top_k to cut noisy context. \
Fragmented chunks also hurt here: given half an explanation, the model \
tends to pad around the gap.

IMPORTANT -- you are given the FULL history of past iterations below. Use it:
  - Increasing top_k tends to help recall but can hurt precision.
  - chunk_size is NOT a one-way lever. Do not assume "smaller is better for \
recall" -- that only holds for fact-dense lookup text. Check the history: if \
recall stalled or fell after a chunk_size REDUCTION, the answers are being \
fragmented and the correct move is to go LARGER, not smaller still.
  - Before concluding chunk_size should shrink, check whether it has ever \
been tried larger. If every past iteration sits in a narrow band well inside \
the permitted bounds, you have not actually tested the parameter -- you have \
been nudging around your starting value. An untested direction is not \
evidence against that direction.
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
values given the history, which MUST cite the document profile numbers behind \
your chunk_size/chunk_overlap choice and the regime you concluded
"""

BALANCED_SEARCH_POLICY = """BALANCED SEARCH POLICY (mandatory):
This is an experiment, not a one-dimensional hill climb. Every proposed
configuration must deliberately consider ALL tunable retrieval dimensions:
chunk_size, chunk_overlap, top_k, retriever_type, and similarity_threshold,
as well as the answer prompt. Do not repeatedly change only top_k and the
threshold. Use the iteration history to maintain an exploration ledger:

1. In the first six non-terminal iterations, make at least one meaningful
   structural change among chunk_size, chunk_overlap, and retriever_type;
   test each of these dimensions at least once when the history has not yet
   tested it. A prompt-only change does not count as structural exploration.
2. Treat context_recall and context_precision as equally important. If either
   is below target, the next trial must explicitly state which one is being
   prioritized and why the other will not be harmed. When precision is below
   target, include a precision-focused trial (lower top_k, MMR, or a stricter
   threshold); when recall is below target, include a recall-focused trial
   (smaller chunks, suitable overlap, or higher top_k).
3. Explore chunk_size and chunk_overlap as a pair: overlap must remain below
   chunk_size and should normally be 10-25% of chunk_size. Do not leave both
   unchanged for more than two consecutive proposals while recall or precision
   misses target.
4. Do not claim a parameter changed unless the JSON value actually changes.
   The JSON configuration is authoritative, not the prose reasoning.
5. Prefer a controlled experiment: change two or three related dimensions at
   most, record the expected metric tradeoff, and avoid repeating a previously
   tested configuration.
"""

DOCUMENT_CONDITIONED_SIZING_POLICY = """DOCUMENT-CONDITIONED SIZING (mandatory):
chunk_size and chunk_overlap must be derived from the DOCUMENT PROFILE above,
not carried over from generic RAG defaults. The profile is not background
colour -- it is data you are required to act on. Two documents of different
size and type must not receive the same chunk settings.

1. Estimate the document's natural unit of meaning:
      chars_per_section = characters / sections
   A chunk should hold a COMPLETE unit of reasoning. Target roughly one
   quarter to one half of chars_per_section for discursive text, or a single
   self-contained entry for reference text. Never size a chunk so small that
   a typical answer must straddle two of them.

2. Pick the regime from doc_type and median_chars_per_page. doc_type is one of
   paper, manual, policy, narrative, unknown:
      * PROSE-HEAVY / ARGUMENTATIVE -- doc_type is paper, narrative, or
        policy, OR median_chars_per_page > 3000. Answers typically span
        several consecutive sentences, so favour LARGER chunks, typically
        800-1200. Do not go below ~600 unless the history shows a specific
        measured gain from doing so.
      * FACT-DENSE / LOOKUP -- doc_type is manual, or doc_type is unknown with
        median_chars_per_page < 2500, OR table_heavy=true, OR list_heavy=true.
        Discrete facts sit in short spans, so favour SMALLER chunks, typically
        300-600.
   If the two signals disagree, say so in your reasoning and pick the regime
   that matches the document summary's actual content.

3. Size chunk_overlap to preserve continuity across a boundary -- enough to
   carry roughly one sentence of lead-in into the next chunk. In practice this
   is 10-20% of chunk_size; larger chunks need proportionally less.

4. Sanity-check against estimated_chunk_count in CURRENT RETRIEVAL SHAPE:
      * Very low (< ~15): top_k is pulling a large fraction of the entire
        document, and precision will suffer.
      * Very high (> ~200) with a small top_k: relevant material is scattered
        across chunks you will never retrieve, and recall will suffer.

5. Your "reasoning" field MUST cite the specific profile numbers you used and
   name the regime you concluded (e.g. "doc_type=paper,
   median_chars_per_page=4353, chars_per_section~3731 -> prose-heavy, so
   chunk_size=1000"). A proposal whose reasoning does not reference the
   profile is not acceptable. If your proposed chunk_size/chunk_overlap would
   be unchanged for a 4-page fact sheet and a 40-page argumentative paper,
   you are ignoring the profile and must revise.
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


def _format_dynamic_context(
    current_config: RetrievalConfig, profile: DocumentProfile
) -> str:
    estimated_count = estimate_chunk_count(
        profile.characters,
        current_config.chunk_size,
        current_config.chunk_overlap,
    )
    # Precomputed so the sizing policy's rule 1 does not depend on the LLM
    # doing arithmetic on the profile numbers itself. Same helpers seed_config.py
    # sizes the starting configuration with, so the shape the LLM is shown and
    # the shape the seed was derived from cannot drift apart.
    return (
        "CURRENT RETRIEVAL SHAPE:\n"
        f"estimated_chunk_count={estimated_count} "
        "(computed from document characters and current chunk settings)\n"
        f"chars_per_section={chars_per_section(profile)} "
        "(characters / sections -- the document's natural unit of meaning; "
        "see DOCUMENT-CONDITIONED SIZING rule 1)"
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
    mistral_settings: MistralSettings,
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

    client = MistralClient(mistral_settings)

    user_message = (
        f"{BALANCED_SEARCH_POLICY}\n\n"
        f"{_format_bounds(opt_config)}\n\n"
        f"DOCUMENT SUMMARY:\n{document_summary}\n\n"
        f"DOCUMENT PROFILE:\n{_format_profile(document_profile)}\n\n"
        f"{_format_dynamic_context(current_config, document_profile)}\n\n"
        # Sizing policy sits immediately after the profile and chunk-count it
        # refers to: the guidance is only actionable next to its own data.
        f"{DOCUMENT_CONDITIONED_SIZING_POLICY}\n\n"
        f"Current configuration: {current_config.to_dict()}\n\n"
        f"Iteration history:\n{_format_history(history)}\n\n"
        "Propose the next configuration."
    )

    raw = client.chat(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        model=mistral_settings.optimizer_model,
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
