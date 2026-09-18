"""
scripts/validate_split_run.py
=============================
End-to-end validation of the cloud/local role split shipped in config/config.yaml.

Runs ONE optimization iteration against the cached Harvey/Abilene benchmark so
every provider role fires on the real pipeline:

    embedding  -> local  (sentence-transformers MiniLM)   builds the FAISS store
    generation -> local  (Ollama qwen2.5:3b)              answers each question
    judge      -> cloud  (Mistral)                         Ragas scoring
    optimizer  -> cloud  (Mistral)                         proposes the next config

It reuses the cached benchmark/summary/profile (no local question-generation
pass) by pointing the path fields at the Harvey artifacts. The FAISS index is
rebuilt because the cached one holds 1024-dim Mistral vectors and MiniLM is
384-dim; that rebuild is local and free. Nothing here mutates config.yaml.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from phoenix_rag.config.loader import load_config, load_env
from phoenix_rag.optimization.runner import run_experiment

ROOT = Path(__file__).resolve().parent.parent
GQ = ROOT / "generated_questions"


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    load_env()
    cfg = load_config(str(ROOT / "config" / "config.yaml"))

    # Point the run at the document that already has a cached benchmark, and at
    # that benchmark, so the local model does no question generation.
    cfg.source_document = str(ROOT / "data" / "Harvey_Abilene_Paradox.pdf")
    cfg.benchmark_path = str(GQ / "benchmark_Harvey_Abilene_Paradox.json")
    cfg.summary_path = str(GQ / "document_summary_Harvey_Abilene_Paradox.txt")
    cfg.profile_path = str(GQ / "document_profile_Harvey_Abilene_Paradox.json")
    cfg.corpus_path = None  # single-document mode

    # A single iteration is enough to prove the split: one local answer pass and
    # one cloud judge + optimizer round.
    cfg.optimizer.max_iterations = 1

    print("=" * 72)
    print("VALIDATION RUN -- cloud/local split")
    print("  source     :", cfg.source_document)
    print("  benchmark  :", cfg.benchmark_path)
    for role in ("embedding", "generation", "optimizer", "judge"):
        p = getattr(cfg.providers, role)
        print(f"  {role:<10} : {p.backend:<8} {p.model}")
    print("=" * 72)

    best = run_experiment(cfg)

    print("\n" + "=" * 72)
    if best:
        print(f"RUN COMPLETE. Best iteration: {best['iteration']}")
        for k, v in best["scores"].items():
            print(f"  {k:<24} {v:.4f}")
    else:
        print("RUN COMPLETE, but no iteration passed the faithfulness gate.")
        print("The split still executed end-to-end; results are in results/.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
