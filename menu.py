"""
menu.py
=======
Terminal front-end for Phoenix RAG. Standard library only -- no new dependency,
so it works anywhere `app.py` already works.

    python menu.py

Every option is a thin wrapper over operations.py, which is also what
streamlit_app.py drives. Nothing decisional lives in this file: it reads input,
prints results, and reports errors back to the menu rather than crashing out of
it, because an operator halfway through assembling a corpus should not lose the
session to a typo in a file path.

    Phoenix RAG
    == corpus: 2 doc(s) | 13 question(s) | best config STALE ==
      1) Optimize RAG
      2) Add document to existing FAISS index
      3) Ask the RAG
      4) Compare old parameters vs re-optimized (new document)
      5) Modify configuration
      6) Show corpus / status
      0) Exit

Options 1 and 4 run in the foreground and can take a long time (they are full
optimization runs against the Mistral API); their progress is the normal log
output, which is why logging goes to stdout here exactly as it does in app.py.
"""

from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv

import operations
import storage
from config import LOGS_DIR, RESULTS_DIR

load_dotenv()

logger = logging.getLogger("phoenix_rag.menu")

RULE = "=" * 68


# =====================================================================
# Input helpers
# =====================================================================

def _setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOGS_DIR / "phoenix_rag.log"),
        ],
    )


def ask_text(prompt: str, default: str | None = None) -> str:
    """Prompt for a line of text. Empty input takes the default, if there is one."""
    suffix = f" [{default}]" if default is not None else ""
    while True:
        try:
            raw = input(f"{prompt}{suffix}: ").strip()
        except EOFError:
            raise KeyboardInterrupt from None
        if raw:
            return raw
        if default is not None:
            return default
        print("  (a value is required)")


def ask_int(prompt: str, default: int) -> int:
    while True:
        raw = ask_text(prompt, str(default))
        try:
            return int(raw)
        except ValueError:
            print(f"  '{raw}' is not a whole number")


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    while True:
        raw = ask_text(f"{prompt} ({hint})", "y" if default else "n").lower()
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("  please answer y or n")


def ask_path(prompt: str, must_exist: bool = True) -> Path:
    while True:
        # Quoted drag-and-drop paths and ~ are both common enough to be worth handling.
        raw = ask_text(prompt).strip().strip("'\"")
        path = Path(raw).expanduser()
        if not must_exist or path.exists():
            return path
        print(f"  no such file: {path}")


# =====================================================================
# 6) Status
# =====================================================================

def show_status(app_config) -> None:
    report = operations.status(app_config)
    print(f"\n{RULE}\nSTATUS\n{RULE}")
    print(f"mode                : {'corpus' if report.corpus_enabled else 'single document'}")
    print(f"benchmark questions : {report.question_count}")
    print(f"active retrieval    : {report.active_retrieval_source}")

    if report.corpus_enabled:
        print(f"corpus root         : {report.corpus_root}")
        print(f"documents           : {report.document_count}")
        for index, document in enumerate(report.documents, start=1):
            profile = document.document_profile
            pages = f"{document.pages}p, " if document.pages else ""
            print(
                f"  {index}. {document.label:<24} {pages}{document.characters:,} chars, "
                f"{profile.doc_type}, {document.question_count} question(s)"
            )
            print(f"     {document.path}")
        if report.index_variants:
            print("index variants      :")
            for variant in report.index_variants:
                print(
                    f"  chunk_size={variant.chunk_size:<6} overlap={variant.chunk_overlap:<5} "
                    f"model={variant.embedding_model:<14} members={len(variant.members)}"
                )
        else:
            print("index variants      : none built yet")
        state = {
            "never": "never optimized against this corpus",
            "stale": "STALE -- membership changed since the last optimization",
            "current": "current -- measured against exactly these documents",
        }
        print(f"best configuration  : {state[report.optimization_status]}")
    else:
        marker = "" if report.source_document_exists else "  (MISSING)"
        print(f"source document     : {report.source_document}{marker}")
        print("corpus              : not enabled (option 2 enables it)")

    if report.best:
        scores = report.best.get("scores", {})
        print(f"best iteration      : {report.best.get('iteration')}")
        print(
            "best scores         : "
            + ", ".join(f"{k}={v:.3f}" for k, v in scores.items())
        )
    else:
        print("best configuration  : none saved yet")

    for problem in report.integrity_problems:
        print(f"!! {problem}")
    print(RULE)


# =====================================================================
# 1) Optimize
# =====================================================================

def optimize(app_config) -> None:
    from experiment_runner import run_experiment

    report = operations.status(app_config)
    print(f"\n{RULE}\nOPTIMIZE RAG\n{RULE}")
    print(f"target        : {report.headline}")
    print(f"benchmark     : {report.question_count} question(s)")
    if report.corpus_enabled and report.document_count > 1:
        print(
            "note          : the optimizer prompt will describe all "
            f"{report.document_count} documents, so the prompt template it writes "
            "must serve every one of them"
        )
    if report.corpus_enabled and report.optimization_status == "stale":
        print(
            "note          : the benchmark grew since the last run, so the scores "
            "below are NOT comparable with the previously saved best"
        )

    iterations = ask_int("max iterations", app_config.optimizer.max_iterations)
    app_config.optimizer.max_iterations = iterations
    if not ask_yes_no(f"Run {iterations} iteration(s) now? This calls the Mistral API"):
        print("cancelled")
        return

    best = run_experiment(app_config)
    if not best:
        print(
            "\nNo iteration passed the faithfulness safety gate, so no best "
            "configuration was saved. Results are still in results/ for inspection."
        )
        return

    print(f"\n{RULE}\nBEST CONFIGURATION (iteration {best['iteration']})\n{RULE}")
    for key, value in best["scores"].items():
        print(f"  {key:<22} {value:.4f}")
    config = best["config"].to_dict()
    for key in ("chunk_size", "chunk_overlap", "top_k", "retriever_type",
                "similarity_threshold"):
        print(f"  {key:<22} {config[key]}")
    print(f"\nsaved to {storage.BEST_CONFIG_PATH}")
    print(RULE)


# =====================================================================
# 2) Add document
# =====================================================================

def add_document(app_config) -> None:
    print(f"\n{RULE}\nADD DOCUMENT TO EXISTING FAISS INDEX\n{RULE}")
    if not app_config.corpus_path:
        print(
            "Corpus mode is off. Adding a document turns it on and seeds the corpus\n"
            f"with the currently configured document:\n  {app_config.source_document}\n"
            "Its cached summary, profile and benchmark are reused, so seeding is free."
        )
        if not ask_yes_no("Continue?"):
            print("cancelled")
            return

    path = ask_path("path to the new document")
    label = ask_text("short label for it", path.stem)
    generate = ask_yes_no(
        "Generate benchmark questions from this document and append them?", True
    )
    if not generate:
        print(
            "  Skipping question generation: this document's chunks will be "
            "retrievable but nothing in the benchmark will test them."
        )

    result = operations.add_document(
        app_config, path=path, label=label, generate_questions=generate
    )
    if result is None:
        print(
            f"\n{path.name} is already in the corpus (identical content). Nothing "
            "to do -- documents are identified by content, not filename."
        )
        return

    print()
    for line in result.describe():
        print(f"  {line}")
    print(
        "\nThe saved best configuration was measured against the smaller benchmark, "
        "so it is now marked stale. Re-run option 1 to tune for the whole corpus."
    )
    print(RULE)


# =====================================================================
# 3) Ask the RAG
# =====================================================================

def ask_the_rag(app_config) -> None:
    print(f"\n{RULE}\nASK THE RAG\n{RULE}")
    print("Loading the index...")
    session = operations.AskSession(app_config)
    config = session.retrieval_config
    print(f"  scope      : {session.scope}")
    print(f"  index      : {session.index_note}")
    print(f"  parameters : {session.provenance}")
    print(
        f"               chunk_size={config.chunk_size}, "
        f"chunk_overlap={config.chunk_overlap}, top_k={config.top_k}, "
        f"retriever={config.retriever_type}"
    )
    print("\nAsk a question, or press Enter on an empty line to return to the menu.")

    while True:
        try:
            question = input("\n? ").strip()
        except EOFError:
            break
        if not question:
            break

        result = session.ask(question)
        print(f"\n{result.answer}")

        attribution = session.attribute(result)
        if attribution:
            print("\n  retrieved from:")
            for label, context in zip(attribution, result.contexts):
                snippet = " ".join(context.split())[:100]
                print(f"    {label}")
                print(f"        {snippet}...")
    print(RULE)


# =====================================================================
# 4) Compare old parameters vs re-optimized
# =====================================================================

def compare(app_config) -> None:
    from document_generalization_experiment import (
        print_comparison,
        run_generalization_experiment,
    )

    print(f"\n{RULE}\nCOMPARE OLD PARAMETERS vs RE-OPTIMIZED\n{RULE}")
    print(
        "Two arms are run on ONE new document:\n"
        "  frozen -- the old document's tuned parameters, applied as-is\n"
        "  fresh  -- the optimizer re-run from scratch on the new document\n"
        "The prompt template is held constant across both arms, so the delta is\n"
        "attributable to the retrieval parameters alone.\n"
        "\nThis is a single-document experiment: it does NOT use the corpus, and it\n"
        "writes to results/generalization_experiment/<label>/ so your top-level\n"
        "results are left alone."
    )

    if not storage.BEST_CONFIG_PATH.exists():
        print(
            f"\nNo saved best configuration at {storage.BEST_CONFIG_PATH} to use as "
            "the frozen arm. Run option 1 first."
        )
        return

    old_best = storage.BEST_CONFIG_PATH
    if ask_yes_no(f"Use a different frozen config than {old_best}?", False):
        old_best = ask_path("old best_configuration.json", must_exist=True)
    new_source = ask_path("path to the NEW document to test on")
    label = ask_text("label for this comparison", new_source.stem)
    iterations = ask_int("max iterations for the fresh arm", 6)

    if not ask_yes_no(
        f"Run both arms now? The fresh arm alone is up to {iterations} iterations "
        "against the Mistral API"
    ):
        print("cancelled")
        return

    result = run_generalization_experiment(
        old_best_config=old_best,
        new_source=new_source,
        label=label,
        max_iterations=iterations,
    )
    print_comparison(
        result["frozen"], result["fresh"], result["control"]["differing_dimensions"]
    )
    print(f"\nfull payload: {result['output_path']}")
    print(f"arm results : {RESULTS_DIR / 'generalization_experiment' / label}")
    print(RULE)


# =====================================================================
# 5) Modify configuration
# =====================================================================

def modify_config(app_config) -> None:
    dirty = False
    while True:
        editable = operations.editable_fields(app_config)
        print(f"\n{RULE}\nCONFIGURATION{'  (unsaved changes)' if dirty else ''}\n{RULE}")
        section = None
        for number, field_ in enumerate(editable, start=1):
            head = field_.path.split(".")[0] if "." in field_.path else "(top level)"
            if head != section:
                section = head
                print(f"\n  [{section}]")
            print(f"   {number:>2}) {field_.path:<44} {field_.display_value}")
        print("\n    s) save to config/default_config.json")
        print("    0) back (discarding unsaved changes)")

        choice = ask_text("\nfield number to edit, or s/0", "0").lower()
        if choice == "0":
            if dirty and not ask_yes_no("Discard unsaved changes?", False):
                continue
            return
        if choice == "s":
            path = operations.save_config(app_config)
            print(f"  saved to {path}")
            dirty = False
            continue

        try:
            index = int(choice)
        except ValueError:
            print(f"  '{choice}' is not one of the listed numbers")
            continue
        # Checked as a range rather than caught as IndexError: index 0 would
        # otherwise resolve to editable[-1] and silently edit the last field.
        if not 1 <= index <= len(editable):
            print(f"  {index} is outside 1-{len(editable)}")
            continue
        field_ = editable[index - 1]

        print(f"\n  {field_.path}")
        if field_.help:
            print(f"  {field_.help}")
        print(f"  kind    : {field_.kind}", end="")
        if field_.choices:
            print(f" ({', '.join(field_.choices)})")
        elif field_.kind in {"int_pair", "float_pair"}:
            print("  -- enter two values, e.g. '2 10'")
        elif field_.kind == "str_list":
            print("  -- comma separated")
        else:
            print()

        if field_.kind == "text":
            print("  current :")
            for line in str(field_.value).splitlines():
                print(f"    {line}")
            print(
                "\n  Enter the new value; finish with a line containing only '.'\n"
                "  (must keep {context} exactly once and {question} exactly once)"
            )
            lines: list[str] = []
            while True:
                try:
                    line = input()
                except EOFError:
                    break
                if line.strip() == ".":
                    break
                lines.append(line)
            if not lines:
                print("  unchanged")
                continue
            raw: object = "\n".join(lines)
        else:
            print(f"  current : {field_.display_value}")
            try:
                raw = input("  new value (Enter to keep): ")
            except EOFError:
                return
            if not raw.strip():
                print("  unchanged")
                continue

        try:
            value = operations.apply_field(app_config, field_.path, raw)
        except operations.ConfigEditError as error:
            print(f"  REJECTED: {error}")
            continue
        dirty = True
        print(f"  {field_.path} = {value!r}  (not saved yet -- press s to save)")


# =====================================================================
# Main loop
# =====================================================================

ACTIONS = {
    "1": ("Optimize RAG", optimize),
    "2": ("Add document to existing FAISS index", add_document),
    "3": ("Ask the RAG", ask_the_rag),
    "4": ("Compare old parameters vs re-optimized (new document)", compare),
    "5": ("Modify configuration", modify_config),
    "6": ("Show corpus / status", show_status),
}


def main() -> int:
    _setup_logging()
    app_config = operations.load_config()

    if not app_config.mistral.api_key:
        print(
            "WARNING: no Mistral API key found. Set MISTRAL_API_KEY in .env, or set\n"
            "mistral.api_key via option 5. Options 1-4 all call the API and will fail\n"
            "without it; options 5 and 6 work offline.\n"
        )

    while True:
        try:
            report = operations.status(app_config)
        except Exception:  # noqa: BLE001 -- never let the header kill the menu
            logger.exception("Could not build the status header")
            header = "status unavailable (see the log)"
        else:
            header = report.headline

        print(f"\n{RULE}\nPhoenix RAG\n== {header} ==")
        for key, (title, _) in ACTIONS.items():
            print(f"  {key}) {title}")
        print("  0) Exit")

        try:
            choice = input("\nchoice: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if choice == "0":
            return 0
        action = ACTIONS.get(choice)
        if action is None:
            print(f"  '{choice}' is not an option")
            continue

        try:
            action[1](app_config)
        except KeyboardInterrupt:
            # Ctrl-C aborts the current operation, not the session -- a long
            # optimization run is exactly the thing an operator wants to be able
            # to stop without losing the corpus they just assembled.
            print("\n  interrupted; returning to the menu")
        except FileNotFoundError as error:
            print(f"\n  {error}")
        except Exception as error:  # noqa: BLE001
            logger.exception("Operation failed")
            print(f"\n  FAILED: {error}")
            print("  (full traceback in the log)")
            if "--debug" in sys.argv:
                traceback.print_exc()


if __name__ == "__main__":
    sys.exit(main())
