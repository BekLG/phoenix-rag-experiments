"""
streamlit_app.py
================
GUI front-end for Phoenix RAG.

    streamlit run streamlit_app.py

Same five operations as menu.py, driven through the same operations.py functions.
Neither front-end contains logic the other lacks -- if the behaviour of "add a
document" needs to change, it changes in operations.py and both UIs follow.

Two Streamlit facts shape this file:

  * The whole script re-runs on every interaction. So the AppConfig lives in
    st.session_state, not in a local -- otherwise unsaved config edits would be
    thrown away by the rerun that follows the next click.
  * A long call blocks the rerun it happens in. Optimization runs therefore show
    a spinner while they work and their captured log afterwards, rather than
    pretending to stream. The log handler is attached once per session and
    accumulates, so the sidebar always has the real progress record.
"""

from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

import corpus
import operations
import storage
from config import LOGS_DIR

load_dotenv()

st.set_page_config(page_title="Phoenix RAG", page_icon="🔥", layout="wide")


# =====================================================================
# Session state
# =====================================================================

def _install_log_handler() -> operations.ListLogHandler:
    """Capture the pipeline's own logging so the GUI can show real progress.

    Attached to the root logger once per session. Every module already logs its
    work in detail; adding a parallel progress-callback mechanism through five
    modules to feed a GUI would be duplicated plumbing for the same information.
    """
    if "log_handler" not in st.session_state:
        handler = operations.ListLogHandler()
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        root.addHandler(handler)
        if not any(isinstance(h, logging.FileHandler) for h in root.handlers):
            file_handler = logging.FileHandler(LOGS_DIR / "phoenix_rag.log")
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
            )
            root.addHandler(file_handler)
        st.session_state.log_handler = handler
    return st.session_state.log_handler


def app_config():
    """The live configuration, surviving reruns so edits are not lost."""
    if "app_config" not in st.session_state:
        st.session_state.app_config = operations.load_config()
    return st.session_state.app_config


def invalidate_session() -> None:
    """Drop the cached AskSession after anything that changes what is indexed."""
    st.session_state.pop("ask_session", None)


def flash(message: str, kind: str = "success") -> None:
    """Queue a message to survive the st.rerun() that follows an action.

    Anything written before st.rerun() is discarded when the script restarts, so
    an action that has to refresh the page (a new document changes the sidebar
    counts and the document table) would otherwise report nothing at all.
    """
    st.session_state.setdefault("_flash", []).append((kind, message))


def render_flashes() -> None:
    for kind, message in st.session_state.pop("_flash", []):
        getattr(st, kind)(message)


log_handler = _install_log_handler()
config = app_config()


# =====================================================================
# Sidebar: status + log
# =====================================================================

with st.sidebar:
    st.markdown("### Status")
    try:
        report = operations.status(config)
    except Exception as error:  # noqa: BLE001 -- the sidebar must not kill the app
        st.error(f"Status unavailable: {error}")
        report = None

    if report is not None:
        st.metric("Mode", "Corpus" if report.corpus_enabled else "Single document")
        columns = st.columns(2)
        columns[0].metric("Documents", report.document_count or 1)
        columns[1].metric("Questions", report.question_count)

        badge = {
            "never": ("Never optimized", st.info),
            "stale": ("Best config STALE", st.warning),
            "current": ("Best config current", st.success),
        }[report.optimization_status]
        badge[1](badge[0])
        st.caption(f"Serving with: {report.active_retrieval_source}")

        for problem in report.integrity_problems:
            st.error(problem)

    if not config.mistral.api_key:
        st.error(
            "No Mistral API key. Set MISTRAL_API_KEY in .env or fill it in on the "
            "Config tab. Everything except the Config tab needs it."
        )

    with st.expander("Run log", expanded=False):
        if log_handler.records:
            st.code("\n".join(log_handler.records[-200:]), language="text")
        else:
            st.caption("Nothing logged yet this session.")
        if st.button("Clear log", use_container_width=True):
            log_handler.records.clear()
            st.rerun()


st.title("Phoenix RAG")
st.caption(report.headline if report is not None else "status unavailable")
render_flashes()

optimize_tab, documents_tab, ask_tab, compare_tab, config_tab = st.tabs(
    ["Optimize", "Documents", "Ask", "Compare", "Config"]
)


# =====================================================================
# Optimize
# =====================================================================

with optimize_tab:
    st.subheader("Optimize the RAG configuration")
    st.write(
        "Runs the full self-optimization loop: the LLM optimizer proposes retrieval "
        "parameters and a prompt template each iteration, they are scored with Ragas "
        "against the fixed benchmark, and the best safe configuration is saved."
    )

    if report is not None and report.corpus_enabled and report.document_count > 1:
        st.info(
            f"The optimizer prompt will describe all {report.document_count} documents, "
            "so the prompt template it writes has to serve every one of them."
        )
    if report is not None and report.optimization_status == "stale":
        st.warning(
            "The benchmark grew since the last run. The new scores are **not** "
            "comparable with the previously saved best -- they are measured against "
            "a different question set."
        )

    iterations = st.number_input(
        "Max iterations",
        min_value=1,
        max_value=50,
        value=int(config.optimizer.max_iterations),
        help="How many configurations the optimizer will try.",
    )

    if st.button("Run optimization", type="primary"):
        from experiment_runner import run_experiment

        config.optimizer.max_iterations = int(iterations)
        with st.spinner(
            f"Running up to {iterations} iteration(s) against the Mistral API. "
            "This takes a while -- the sidebar log updates as each step completes."
        ):
            try:
                best = run_experiment(config)
            except Exception as error:  # noqa: BLE001
                logging.getLogger("phoenix_rag.streamlit").exception("Optimization failed")
                st.error(f"Optimization failed: {error}")
                best = None
            else:
                invalidate_session()

        if best:
            st.success(f"Best configuration found at iteration {best['iteration']}")
            score_columns = st.columns(len(best["scores"]) or 1)
            for column, (name, value) in zip(score_columns, best["scores"].items()):
                column.metric(name, f"{value:.3f}")
            st.json(best["config"].to_dict())
            st.caption(f"Saved to {storage.BEST_CONFIG_PATH}")
        elif best is not None:
            st.warning(
                "No iteration passed the faithfulness safety gate, so nothing was "
                "saved as best. The per-iteration results are still under results/."
            )

    if report is not None and report.best:
        with st.expander("Currently saved best configuration"):
            st.json(report.best)


# =====================================================================
# Documents
# =====================================================================

with documents_tab:
    st.subheader("Documents in the index")

    if report is not None and not report.corpus_enabled:
        st.info(
            "Corpus mode is off. Adding a document turns it on and seeds the corpus "
            f"with the configured document (`{Path(report.source_document).name}`), "
            "reusing its cached summary, profile and benchmark so seeding costs nothing."
        )

    if report is not None and report.documents:
        st.dataframe(
            [
                {
                    "label": document.label,
                    "pages": document.pages or "-",
                    "characters": document.characters,
                    "doc_type": document.document_profile.doc_type,
                    "questions": document.question_count,
                    "added": document.added_at[:19].replace("T", " "),
                    "path": document.path,
                }
                for document in report.documents
            ],
            use_container_width=True,
            hide_index=True,
        )
        if report.index_variants:
            with st.expander("FAISS index variants"):
                st.dataframe(
                    [
                        {
                            "chunk_size": variant.chunk_size,
                            "chunk_overlap": variant.chunk_overlap,
                            "embedding_model": variant.embedding_model,
                            "documents embedded": len(variant.members),
                            "key": variant.index_key,
                        }
                        for variant in report.index_variants
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption(
                    "One index per (embedding model, chunk_size, chunk_overlap). "
                    "Adding a document extends every variant it is missing from; "
                    "changing chunk_size necessarily builds a new one, because "
                    "differently-sized chunks are different vectors."
                )

    st.divider()
    st.markdown("#### Add a document to the existing FAISS index")
    st.write(
        "Only the new document's chunks are embedded -- the vectors already in the "
        "index are reused. Its summary and profile join the existing ones, so the "
        "optimizer prompt reasons about every document's concepts at once."
    )

    upload = st.file_uploader(
        "Document", type=["pdf", "txt", "md", "markdown"], accept_multiple_files=False
    )
    existing_path = st.text_input(
        "...or a path already on disk",
        placeholder="data/some_document.pdf",
        help="Used when nothing is uploaded above.",
    )

    default_label = ""
    if upload is not None:
        default_label = Path(upload.name).stem
    elif existing_path.strip():
        default_label = Path(existing_path.strip()).stem

    label = st.text_input("Label", value=default_label)
    generate = st.checkbox(
        "Generate benchmark questions from this document and append them",
        value=True,
        help=(
            "Without this, the document is retrievable but nothing in the benchmark "
            "tests it, so optimizing would tune the parameters to ignore it."
        ),
    )

    if st.button("Add document", type="primary", disabled=upload is None and not existing_path.strip()):
        try:
            if upload is not None:
                source = operations.stage_document(upload.name, upload.getvalue())
            else:
                source = Path(existing_path.strip()).expanduser()
            with st.spinner(f"Profiling, summarizing and embedding {source.name}..."):
                result = operations.add_document(
                    config,
                    path=source,
                    label=label or source.stem,
                    generate_questions=generate,
                )
        except FileNotFoundError as error:
            st.error(str(error))
        except Exception as error:  # noqa: BLE001
            logging.getLogger("phoenix_rag.streamlit").exception("Add document failed")
            st.error(f"Could not add the document: {error}")
        else:
            invalidate_session()
            if result is None:
                flash(
                    f"{source.name} is already in the corpus -- identical content. "
                    "Documents are identified by content, not filename, so this was "
                    "a no-op.",
                    "warning",
                )
            else:
                flash(
                    f"Added **{result.document.label}**\n\n"
                    + "\n".join(f"- {line}" for line in result.describe())
                )
                flash(
                    "The saved best configuration was measured against the smaller "
                    "benchmark and is now stale. Re-run Optimize to tune for the "
                    "whole corpus.",
                    "warning",
                )
            st.rerun()

    if report is not None and report.documents:
        st.divider()
        with st.expander("Remove a document"):
            st.caption(
                "FAISS has no cheap per-vector removal, so removing a document "
                "invalidates every index variant that contained it: the next run "
                "rebuilds from the remaining documents. Its questions stay in the "
                "benchmark unless you regenerate it."
            )
            victim = st.selectbox(
                "Document", [document.label for document in report.documents]
            )
            if st.button("Remove", type="secondary"):
                removed = operations.remove_document(config, victim)
                invalidate_session()
                if removed is None:
                    st.error(f"No document labelled {victim}")
                else:
                    flash(f"Removed {removed.label}")
                    st.rerun()

    if report is not None and report.corpus_enabled and report.documents:
        with st.expander("Summary the optimizer will see"):
            st.code(corpus.summary_text(operations.corpus_state(config)), language="text")


# =====================================================================
# Ask
# =====================================================================

with ask_tab:
    st.subheader("Ask the RAG")

    if "ask_session" not in st.session_state:
        st.write(
            "Loads the index for the currently active retrieval parameters, then "
            "answers questions through the same pipeline the optimizer scores."
        )
        if st.button("Load index", type="primary"):
            with st.spinner("Loading (or extending) the FAISS index..."):
                try:
                    st.session_state.ask_session = operations.AskSession(config)
                except Exception as error:  # noqa: BLE001
                    logging.getLogger("phoenix_rag.streamlit").exception("Load failed")
                    st.error(f"Could not load the index: {error}")
                else:
                    st.rerun()

    session = st.session_state.get("ask_session")
    if session is not None:
        retrieval = session.retrieval_config
        info_columns = st.columns(3)
        info_columns[0].caption(f"**Scope**\n\n{session.scope}")
        info_columns[1].caption(f"**Parameters from**\n\n{session.provenance}")
        info_columns[2].caption(
            f"**Retrieval**\n\nchunk_size={retrieval.chunk_size}, "
            f"overlap={retrieval.chunk_overlap}, top_k={retrieval.top_k}, "
            f"{retrieval.retriever_type}"
        )
        st.caption(session.index_note)
        if st.button("Reload index"):
            invalidate_session()
            st.rerun()

        st.session_state.setdefault("qa_history", [])
        # A form rather than st.chat_input: chat_input is restricted to the main
        # body in several Streamlit versions and raises inside st.tabs.
        with st.form("ask_form", clear_on_submit=True):
            question = st.text_input(
                "Question", placeholder="Ask about the indexed document(s)"
            )
            asked_now = st.form_submit_button("Ask", type="primary")

        if asked_now and question.strip():
            with st.spinner("Retrieving and generating..."):
                try:
                    result = session.ask(question.strip())
                except Exception as error:  # noqa: BLE001
                    logging.getLogger("phoenix_rag.streamlit").exception("Ask failed")
                    st.error(f"Query failed: {error}")
                else:
                    st.session_state.qa_history.append(
                        (question.strip(), result.answer,
                         list(zip(session.attribute(result), result.contexts)))
                    )

        for asked, answer, contexts in reversed(st.session_state.qa_history):
            with st.chat_message("user"):
                st.write(asked)
            with st.chat_message("assistant"):
                st.write(answer)
                with st.expander(f"Retrieved context ({len(contexts)} chunk(s))"):
                    for attribution, context in contexts:
                        st.markdown(f"**{attribution}**")
                        st.text(context)


# =====================================================================
# Compare
# =====================================================================

with compare_tab:
    st.subheader("Compare old parameters vs re-optimized")
    st.write(
        "Two arms on **one new document**: *frozen* reuses the old document's tuned "
        "parameters as-is, *fresh* re-runs the optimizer from scratch on the new "
        "document. The prompt template is held constant across both, so the delta is "
        "attributable to the retrieval parameters alone."
    )
    st.info(
        "This is inherently a single-document experiment -- it does not use the "
        "corpus, and it writes under results/generalization_experiment/<label>/ so "
        "your top-level results are untouched."
    )

    frozen_source = st.text_input(
        "Frozen configuration (old best_configuration.json)",
        value=str(storage.BEST_CONFIG_PATH),
    )
    compare_upload = st.file_uploader(
        "New document to test on",
        type=["pdf", "txt", "md", "markdown"],
        key="compare_upload",
    )
    compare_path = st.text_input(
        "...or a path already on disk", key="compare_path", placeholder="data/other.pdf"
    )
    compare_label = st.text_input(
        "Label for this comparison",
        value=(
            Path(compare_upload.name).stem
            if compare_upload is not None
            else Path(compare_path.strip()).stem if compare_path.strip() else ""
        ),
        help="Namespaces this document's benchmark, summary, profile and results.",
    )
    compare_iterations = st.number_input(
        "Max iterations for the fresh arm", min_value=1, max_value=30, value=6
    )

    ready = (compare_upload is not None or bool(compare_path.strip())) and bool(
        compare_label.strip()
    )
    if st.button("Run comparison", type="primary", disabled=not ready):
        from document_generalization_experiment import run_generalization_experiment

        try:
            if compare_upload is not None:
                new_source = operations.stage_document(
                    compare_upload.name, compare_upload.getvalue()
                )
            else:
                new_source = Path(compare_path.strip()).expanduser()
            with st.spinner(
                "Running the fresh arm, then the frozen arm. This is the longest "
                "operation here -- watch the sidebar log."
            ):
                comparison = run_generalization_experiment(
                    old_best_config=frozen_source,
                    new_source=new_source,
                    label=compare_label.strip(),
                    max_iterations=int(compare_iterations),
                )
        except FileNotFoundError as error:
            st.error(str(error))
        except Exception as error:  # noqa: BLE001
            logging.getLogger("phoenix_rag.streamlit").exception("Comparison failed")
            st.error(f"Comparison failed: {error}")
        else:
            st.session_state.comparison = comparison

    comparison = st.session_state.get("comparison")
    if comparison:
        frozen, fresh = comparison["frozen"], comparison["fresh"]
        differing = comparison["control"]["differing_dimensions"]

        if not differing:
            st.warning(
                "The two arms are identical on every compared dimension, so the delta "
                "below is a run-to-run noise estimate, not an effect."
            )
        else:
            st.markdown("**Parameters that differ**")
            st.dataframe(
                [
                    {"dimension": dimension, "frozen": str(values["frozen"]),
                     "fresh": str(values["fresh"])}
                    for dimension, values in differing.items()
                ],
                use_container_width=True,
                hide_index=True,
            )

        metrics = sorted(set(frozen.get("scores", {})) | set(fresh.get("scores", {})))
        st.markdown("**Scores**")
        st.dataframe(
            [
                {
                    "metric": metric,
                    "frozen": round(frozen.get("scores", {}).get(metric, 0.0), 4),
                    "fresh": round(fresh.get("scores", {}).get(metric, 0.0), 4),
                    "delta (fresh - frozen)": round(
                        fresh.get("scores", {}).get(metric, 0.0)
                        - frozen.get("scores", {}).get(metric, 0.0),
                        4,
                    ),
                }
                for metric in metrics
            ],
            use_container_width=True,
            hide_index=True,
        )
        st.caption(f"Full payload: {comparison['output_path']}")
        with st.expander("Prompt template held constant across both arms"):
            st.code(comparison["control"]["shared_prompt_template"], language="text")


# =====================================================================
# Config
# =====================================================================

with config_tab:
    st.subheader("Configuration")
    st.caption(
        "Edits apply to this session immediately and are written to "
        "config/default_config.json only when you save."
    )

    editable = operations.editable_fields(config)
    sections: dict[str, list] = {}
    for field_ in editable:
        head = field_.path.split(".")[0] if "." in field_.path else "general"
        sections.setdefault(head, []).append(field_)

    with st.form("config_form"):
        updates: dict[str, object] = {}
        # What each widget was SEEDED with, so "did the user change this?" compares
        # like with like below. Pair and list fields are rendered as a joined
        # string, so comparing the widget's "2, 10" against the stored (2, 10)
        # would report every one of them as edited on every submit.
        rendered: dict[str, object] = {}
        for section, section_fields in sections.items():
            with st.expander(section, expanded=section in {"retrieval", "question_generation"}):
                for field_ in section_fields:
                    key = f"cfg::{field_.path}"
                    name = field_.path.split(".")[-1]
                    help_text = field_.help or None
                    rendered[field_.path] = (
                        ", ".join(str(v) for v in field_.value)
                        if field_.kind in {"int_pair", "float_pair", "str_list"}
                        else field_.value
                    )

                    if field_.kind == "bool":
                        updates[field_.path] = st.checkbox(
                            name, value=bool(field_.value), key=key, help=help_text
                        )
                    elif field_.kind == "choice":
                        options = list(field_.choices or ())
                        index = options.index(field_.value) if field_.value in options else 0
                        updates[field_.path] = st.selectbox(
                            name, options, index=index, key=key, help=help_text
                        )
                    elif field_.kind == "int":
                        updates[field_.path] = st.number_input(
                            name, value=int(field_.value), step=1, key=key, help=help_text
                        )
                    elif field_.kind == "float":
                        updates[field_.path] = st.number_input(
                            name,
                            value=float(field_.value),
                            step=0.05,
                            format="%.4f",
                            key=key,
                            help=help_text,
                        )
                    elif field_.kind in {"int_pair", "float_pair", "str_list"}:
                        updates[field_.path] = st.text_input(
                            name,
                            value=", ".join(str(v) for v in field_.value),
                            key=key,
                            help=(help_text or "")
                            + (" Two values, low first." if field_.kind.endswith("pair")
                               else " Comma separated."),
                        )
                    elif field_.kind == "text":
                        updates[field_.path] = st.text_area(
                            name, value=str(field_.value), height=260, key=key,
                            help=help_text,
                        )
                    elif field_.secret:
                        # Left blank on load and skipped when still blank, so the
                        # form cannot silently wipe a key it never displayed.
                        entered = st.text_input(
                            name, value="", type="password", key=key,
                            placeholder=field_.display_value,
                            help="Leave blank to keep the current value.",
                        )
                        if entered.strip():
                            updates[field_.path] = entered.strip()
                    else:
                        updates[field_.path] = st.text_input(
                            name, value=str(field_.value), key=key, help=help_text
                        )

        columns = st.columns(2)
        applied = columns[0].form_submit_button("Apply to this session")
        saved = columns[1].form_submit_button("Apply and save", type="primary")

    if applied or saved:
        # Only genuinely changed fields are submitted: passing every field through
        # coercion each time would turn a tuple into a list on a no-op submit, and
        # would fight the secret-field rule above.
        changed = {
            path: value
            for path, value in updates.items()
            if str(value) != str(rendered[path])
        }
        if not changed:
            st.info("Nothing changed.")
        else:
            try:
                operations.apply_fields(config, changed)
            except operations.ConfigEditError as error:
                st.error(f"Rejected: {error}")
            else:
                invalidate_session()
                verb = "Applied and saved" if saved else "Applied to this session"
                if saved:
                    operations.save_config(config)
                flash(f"{verb}: {', '.join(sorted(changed))}")
                st.rerun()

    st.caption(
        "Changing `retrieval.chunk_size` or `chunk_overlap` means the next run "
        "builds a new index variant -- differently-sized chunks are different "
        "vectors and cannot be reused."
    )
