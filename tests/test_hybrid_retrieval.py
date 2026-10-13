"""Offline Phase 2 tests: real BM25/FAISS where installed, fake embeddings/LLM."""

import builtins
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from unittest import mock, skipIf

import pytest
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from phoenix_rag import storage
from phoenix_rag.config import AppConfig, RetrievalConfig, load_config, save_config
from phoenix_rag.core import corpus
from phoenix_rag.core.chunking import split_documents
from phoenix_rag.core.rag_pipeline import RagPipeline
from phoenix_rag.core.vector_store import add_chunks_to_store
from phoenix_rag.providers.base import ChatProvider
from phoenix_rag.retrieval import build_retriever
from phoenix_rag.retrieval.base import RetrievalResult
from phoenix_rag.retrieval.bm25 import BM25Retriever, tokenize
from phoenix_rag.retrieval.dense import DenseRetriever
from phoenix_rag.retrieval.hybrid import HybridRetriever

from test_corpus import FAISS_AVAILABLE, FakeEmbeddings
from test_retrieval import FakeStore, stored_documents

BM25_AVAILABLE = importlib.util.find_spec("rank_bm25") is not None
needs_bm25 = skipIf(not BM25_AVAILABLE, "rank_bm25 is not installed")

TEXTS = [
    "A quasar emits radiation from a distant galactic core.",
    "Volcanic magma cools into basalt rock.",
    "Honeybees carry pollen between spring flowers.",
    "Penguins survive cold Antarctic winters.",
    "A telescope observes nebula clouds and distant stars.",
    "Sourdough bread ferments with yeast cultures.",
]


@pytest.fixture
def tiny_store():
    if not FAISS_AVAILABLE:
        pytest.skip("faiss-cpu is not installed")
    chunks = split_documents([
        Document(page_content=text, metadata={"doc_id": f"document-{i}"})
        for i, text in enumerate(TEXTS)
    ], 100, 0)
    return FAISS.from_documents(chunks, FakeEmbeddings())


def test_tokenizer_is_deterministic_unicode_aware_and_shared():
    text = "CAFÉ, naïve! አዲስ አበባ 中文 — co-op 42_under."
    expected = ["café", "naïve", "አዲስ", "አበባ", "中文", "co", "op", "42_under"]
    assert tokenize(text) == tokenize(text) == tokenize(text.lower()) == expected
    assert tokenize("... \n\t") == []


@needs_bm25
def test_bm25_lexical_ids_scores_and_same_canonical_corpus_as_dense(tiny_store):
    from rank_bm25 import BM25Okapi

    config = RetrievalConfig(top_k=len(TEXTS), chunk_size=100, chunk_overlap=0)
    docs = stored_documents(tiny_store)
    with mock.patch("phoenix_rag.retrieval.bm25.tokenize", wraps=tokenize) as tokenizer:
        bm25 = BM25Retriever(tiny_store, config)
        results = bm25.retrieve("QUASAR!")
    assert tokenizer.call_args_list == [mock.call(d.page_content) for d in docs] + [mock.call("QUASAR!")]
    assert results[0].chunk_id == docs[0].metadata["chunk_id"]
    assert results[0].retrieval_score > 0
    expected_scores = BM25Okapi([tokenize(d.page_content) for d in docs]).get_scores(["quasar"])
    score_by_id = {d.metadata["chunk_id"]: float(s) for d, s in zip(docs, expected_scores)}
    assert {r.chunk_id: r.retrieval_score for r in results} == score_by_id
    dense = DenseRetriever(tiny_store, config).retrieve("QUASAR!")
    assert {r.chunk_id for r in results} == {r.chunk_id for r in dense}
    assert {r.document_id for r in results} == {f"document-{i}" for i in range(len(TEXTS))}
    assert [r.rank for r in results] == list(range(1, len(TEXTS) + 1))
    assert all(r.retrieval_method == "bm25" and r.extras == {} for r in results)
    assert results[0].metadata == docs[0].metadata
    assert results[0].metadata is not docs[0].metadata


@needs_bm25
def test_bm25_builds_once_across_queries_and_pipeline_configs(tiny_store):
    from rank_bm25 import BM25Okapi

    config = RetrievalConfig(retrieval_strategy="bm25", top_k=2)
    with mock.patch("rank_bm25.BM25Okapi", wraps=BM25Okapi) as constructor:
        retriever = build_retriever(config, tiny_store)
        for query in ["quasar", "magma", "pollen"] * 4:
            assert len(retriever.retrieve(query)) == 2
        # New pipeline iterations change retrieval settings but reuse this store.
        for strategy in ("bm25", "hybrid"):
            other = build_retriever(config.copy_with(
                retrieval_strategy=strategy, top_k=3, hybrid_dense_k=4,
                hybrid_bm25_k=5, rrf_k=20,
            ), tiny_store)
            assert len(other.retrieve("quasar")) == 3
        assert constructor.call_count == 1


@needs_bm25
def test_bm25_rebuilds_full_index_on_in_place_growth_replacement_and_chunking(tiny_store):
    from rank_bm25 import BM25Okapi

    config = RetrievalConfig(top_k=20, chunk_size=100, chunk_overlap=0)
    with mock.patch("rank_bm25.BM25Okapi", wraps=BM25Okapi) as constructor:
        bm25 = BM25Retriever(tiny_store, config)
        old_ids = {r.chunk_id for r in bm25.retrieve("quasar")}
        new = split_documents([Document(
            page_content="Axolotl regeneration restores limbs.", metadata={"doc_id": "added"},
        )], 100, 0)
        add_chunks_to_store(tiny_store, new)
        result = bm25.retrieve("axolotl")
        assert result[0].chunk_id == new[0].metadata["chunk_id"]
        assert {r.chunk_id for r in result} == old_ids | {new[0].metadata["chunk_id"]}
        assert constructor.call_count == 2
        assert len(constructor.call_args.args[0]) == len(TEXTS) + 1
        bm25.retrieve("quasar")
        assert constructor.call_count == 2

        # Membership changed but its size did not: count alone cannot be a key.
        tiny_store.delete([tiny_store.index_to_docstore_id[0]])
        replacement = split_documents([Document(
            page_content="Zircon crystals preserve geological history.", metadata={"doc_id": "replacement"},
        )], 100, 0)
        add_chunks_to_store(tiny_store, replacement)
        result = bm25.retrieve("zircon")
        assert result[0].chunk_id == replacement[0].metadata["chunk_id"]
        assert len(result) == len(TEXTS) + 1
        assert constructor.call_count == 3
        for changed in (config.copy_with(chunk_size=80), config.copy_with(chunk_overlap=5)):
            BM25Retriever(tiny_store, changed).retrieve("zircon")
        assert constructor.call_count == 5


@needs_bm25
@skipIf(not FAISS_AVAILABLE, "faiss-cpu is not installed")
def test_incremental_corpus_sync_load_and_rechunk_share_all_ids(tmp_path):
    state = corpus.Corpus(root=tmp_path / "corpus")
    embeddings = FakeEmbeddings()

    def add_document(label, paragraphs):
        path = tmp_path / f"{label}.txt"
        path.write_text("\n\n".join(paragraphs), encoding="utf-8")
        state.documents.append(corpus.CorpusDocument(
            doc_id=corpus.document_digest(path), label=label, path=str(path),
            pages=None, characters=len(path.read_text()), added_at="", profile={}, summary="",
        ))

    def sync(size, overlap):
        return corpus.sync_index(state, embeddings, "fake-embedding", size, overlap)

    add_document("first", TEXTS[:3])
    initial, built = sync(100, 0)
    initial_ids = {r.chunk_id for r in BM25Retriever(initial, RetrievalConfig(top_k=50)).retrieve("quasar")}
    embeddings.embedded_texts.clear()
    add_document("second", TEXTS[3:])
    grown, extended = sync(100, 0)
    assert extended.action == "extended"
    assert extended.index_key == built.index_key
    assert embeddings.call_count == extended.added_chunks

    loaded, reused = sync(100, 0)
    rechunked, changed = sync(30, 5)
    assert reused.action == "loaded"
    assert changed.index_key != built.index_key
    for store, size, overlap in ((grown, 100, 0), (loaded, 100, 0), (rechunked, 30, 5)):
        config = RetrievalConfig(top_k=100, chunk_size=size, chunk_overlap=overlap)
        dense = DenseRetriever(store, config).retrieve("penguins")
        lexical = BM25Retriever(store, config).retrieve("penguins")
        assert {r.chunk_id for r in dense} == {r.chunk_id for r in lexical}
        assert {r.document_id for r in lexical} == set(state.document_ids)
        assert "Penguins" in lexical[0].text
        if size == 100:
            assert initial_ids < {r.chunk_id for r in lexical}


@needs_bm25
@pytest.mark.parametrize("query", ["unknownword", "!!!"])
def test_bm25_zero_score_ties_are_deterministic(tiny_store, query):
    results = BM25Retriever(tiny_store, RetrievalConfig(top_k=20)).retrieve(query)
    assert [r.chunk_id for r in results] == sorted(r.chunk_id for r in results)
    assert all(r.retrieval_score == 0 for r in results)


@needs_bm25
@pytest.mark.parametrize("texts", [[], ["...", "!!!"]])
def test_bm25_handles_empty_and_tokenless_corpus(texts):
    store = FakeStore()
    for i, text in enumerate(texts):
        store.docs[i].page_content = text
    store.index_to_docstore_id = {i: str(i) for i in range(len(texts))}
    results = BM25Retriever(store, RetrievalConfig(top_k=10)).retrieve("query")
    assert len(results) == len(texts)
    assert all(r.retrieval_score == 0 for r in results)


def hit(chunk_id, rank, score=9999.0):
    return RetrievalResult(chunk_id, "document", f"text {chunk_id}", {"chunk_id": chunk_id},
                           rank, score, "source")


def stub(results):
    return mock.Mock(retrieve=mock.Mock(return_value=results))


def test_hand_computed_rrf_merges_overlap_and_preserves_single_list_hits():
    dense = stub([hit("a", 1, -100), hit("b", 2, 0.1), hit("c", 3, 10000)])
    bm25 = stub([hit("b", 1, 700), hit("d", 2, -1), hit("a", 3, 0)])
    results = HybridRetriever(dense, bm25, top_k=4, rrf_k=60).retrieve("question")
    assert [r.chunk_id for r in results] == ["b", "a", "d", "c"]
    assert [r.rank for r in results] == [1, 2, 3, 4]
    expected = [
        (1 / 62 + 1 / 61, 2, 1),
        (1 / 61 + 1 / 63, 1, 3),
        (1 / 62, None, 2),
        (1 / 63, 3, None),
    ]
    for result, (score, dense_rank, bm25_rank) in zip(results, expected):
        assert result.retrieval_method == "hybrid"
        assert result.retrieval_score == score
        assert result.extras == {"dense_rank": dense_rank, "bm25_rank": bm25_rank, "rrf_score": score}
    dense.retrieve.assert_called_once_with("question")
    bm25.retrieve.assert_called_once_with("question")
    assert dense.retrieve.return_value[0].retrieval_method == "source"
    assert dense.retrieve.return_value[0].extras == {}
    # Input scores are arbitrary and irrelevant to fusion.
    dense.retrieve.return_value = [replace(r, retrieval_score=None) for r in dense.retrieve.return_value]
    bm25.retrieve.return_value = [replace(r, retrieval_score=-999) for r in bm25.retrieve.return_value]
    assert HybridRetriever(dense, bm25, 4).retrieve("question") == results


def test_rrf_ties_use_best_rank_then_chunk_id_independent_of_insertion_order():
    # With k=1, 1/3 + 1/6 == 1/2. The rank-1 hit wins before ID sorting.
    dense_hits = [hit("a", 2), hit("z", 1), hit("b", 3)]
    lexical_hits = [hit("a", 5), hit("c", 3)]
    for dense in (dense_hits, list(reversed(dense_hits))):
        for lexical in (lexical_hits, list(reversed(lexical_hits))):
            results = HybridRetriever(stub(dense), stub(lexical), 4, rrf_k=1).retrieve("q")
            assert [r.chunk_id for r in results] == ["z", "a", "b", "c"]
            assert results[0].retrieval_score == results[1].retrieval_score == 0.5
            assert results[2].retrieval_score == results[3].retrieval_score == 0.25


@pytest.mark.parametrize("dense_hits,bm25_hits,expected", [
    ([], [], []),
    ([hit("a", 1)], [], ["a"]),
    ([], [hit("b", 1)], ["b"]),
    ([hit("a", 2), hit("a", 1)], [hit("a", 1)], ["a"]),
])
def test_rrf_empty_lists_and_repeated_ids(dense_hits, bm25_hits, expected):
    results = HybridRetriever(stub(dense_hits), stub(bm25_hits), 3).retrieve("q")
    assert [r.chunk_id for r in results] == expected
    if dense_hits and bm25_hits:
        assert results[0].retrieval_score == 2 / 61
        assert results[0].extras == {"dense_rank": 1, "bm25_rank": 1, "rrf_score": 2 / 61}


def test_rrf_truncates_after_fusion_and_assigns_fresh_ranks():
    results = HybridRetriever(stub([hit("a", 1), hit("b", 2)]),
                              stub([hit("b", 1), hit("c", 2)]), 1).retrieve("q")
    assert [(r.chunk_id, r.rank) for r in results] == [("b", 1)]


@pytest.mark.parametrize("method", ["similarity", "mmr", "similarity_score_threshold"])
def test_default_factory_preserves_phase1_dense_results(method):
    store = FakeStore()
    config = RetrievalConfig(retriever_type=method, top_k=3, similarity_threshold=0.5)
    retriever = build_retriever(config, store)
    assert isinstance(retriever, DenseRetriever)
    assert retriever.retrieve("q") == DenseRetriever(store, config).retrieve("q")


@needs_bm25
@pytest.mark.parametrize("strategy,expected", [
    ("dense", DenseRetriever), ("bm25", BM25Retriever), ("hybrid", HybridRetriever),
])
def test_factory_strategy_classes(strategy, expected):
    assert isinstance(build_retriever(RetrievalConfig(retrieval_strategy=strategy), FakeStore()), expected)


@needs_bm25
@pytest.mark.parametrize("top_k,dense_k,bm25_k,expected", [
    (2, None, None, (10, 10)), (7, None, None, (14, 14)),
    (2, 3, 5, (3, 5)), (2, None, 1, (10, 1)),
])
def test_hybrid_candidate_budgets_preserve_config(top_k, dense_k, bm25_k, expected):
    store = FakeStore()
    config = RetrievalConfig(retrieval_strategy="hybrid", top_k=top_k,
                             hybrid_dense_k=dense_k, hybrid_bm25_k=bm25_k)
    before = config.to_dict()
    retriever = build_retriever(config, store)
    assert retriever._dense._retriever.search_kwargs["k"] == expected[0]
    assert retriever._bm25._top_k == expected[1]
    assert len(retriever.retrieve("alpha")) <= top_k
    assert config.to_dict() == before


@needs_bm25
@pytest.mark.parametrize("strategy", ["dense", "bm25", "hybrid"])
def test_pipeline_end_to_end_with_fake_llm_and_debug_logging(tiny_store, strategy, caplog):
    generation = mock.Mock(spec=ChatProvider)
    generation.chat.return_value = "An offline generated answer."
    config = RetrievalConfig(retrieval_strategy=strategy, top_k=2)
    pipeline = RagPipeline(tiny_store, generation, config)
    with caplog.at_level("DEBUG", logger="phoenix_rag.retrieval"):
        result = pipeline.answer("What emits quasar radiation?")
    assert result.answer == "An offline generated answer."
    assert len(result.contexts) == 2
    assert [r.rank for r in result.retrieval_results] == [1, 2]
    assert result.contexts == [r.text for r in result.retrieval_results]
    assert result.sources == [r.metadata for r in result.retrieval_results]
    assert [d.page_content for d in pipeline.retrieve(result.question)] == result.contexts
    generation.chat.assert_called_once_with(messages=[{
        "role": "user", "content": config.prompt_template.format(
            context="\n\n".join(result.contexts), question=result.question,
        ),
    }], temperature=0.2)
    assert f"strategy={strategy}" in caplog.text
    assert "latency_ms=" in caplog.text and "candidates=" in caplog.text
    assert "final_chunk_ids=" in caplog.text
    assert all(r.chunk_id in caplog.text for r in result.retrieval_results)
    if strategy == "hybrid":
        assert "dense_top_ids=" in caplog.text and "bm25_top_ids=" in caplog.text
        assert "fused_top_ids=" in caplog.text


@needs_bm25
@pytest.mark.parametrize("method", ["mmr", "similarity_score_threshold"])
def test_dense_method_is_respected_in_hybrid_and_ignored_by_bm25(method):
    store = FakeStore()
    config = RetrievalConfig(retrieval_strategy="bm25", top_k=3,
                             retriever_type=method, similarity_threshold=0.95)
    bm25 = build_retriever(config, store).retrieve("alpha")
    assert len(bm25) == 3
    assert store.calls == []
    hybrid = build_retriever(config.copy_with(retrieval_strategy="hybrid"), store)
    hybrid.retrieve("alpha")
    assert hybrid._dense._retriever.search_type == method
    assert len(store.calls) == 1


@pytest.mark.parametrize("field", ["top_k", "hybrid_dense_k", "hybrid_bm25_k", "rrf_k"])
@pytest.mark.parametrize("value", [0, -1])
def test_config_rejects_nonpositive_budgets(field, value):
    with pytest.raises(ValueError, match=f"{field} must be a positive integer"):
        RetrievalConfig.from_dict({field: value})


@pytest.mark.parametrize("value", ["2", 1.5, True, None])
def test_config_rejects_invalid_top_k_types(value):
    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        RetrievalConfig(top_k=value)


def test_unknown_strategy_and_mutated_config_fail_clearly():
    with pytest.raises(ValueError, match="Unknown retrieval_strategy.*unknown.*dense.*bm25.*hybrid"):
        RetrievalConfig.from_dict({"retrieval_strategy": "unknown"})
    config = RetrievalConfig()
    config.retrieval_strategy = "unknown"
    with pytest.raises(ValueError, match="Unknown retrieval_strategy"):
        build_retriever(config, FakeStore())


def test_old_config_loads_new_defaults(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({"retrieval": {"top_k": 3, "retriever_type": "mmr"}}))
    config = load_config(path).retrieval
    assert config.retrieval_strategy == "dense"
    assert config.hybrid_dense_k is config.hybrid_bm25_k is None
    assert config.rrf_k == 60
    assert config.top_k == 3 and config.retriever_type == "mmr"


@pytest.mark.parametrize("budgets", [(None, None), (7, 9)])
def test_config_copy_yaml_json_and_iteration_storage_preserve_fields(tmp_path, budgets):
    config = RetrievalConfig(retrieval_strategy="hybrid", hybrid_dense_k=budgets[0],
                             hybrid_bm25_k=budgets[1], rrf_k=20)
    copied = config.copy_with(top_k=4, chunk_size=500)
    assert copied.to_dict() == {**config.to_dict(), "top_k": 4, "chunk_size": 500}
    app = AppConfig(retrieval=copied)
    save_config(app, tmp_path / "config.yaml")
    assert load_config(tmp_path / "config.yaml").retrieval == copied
    (tmp_path / "config.json").write_text(json.dumps(app.to_dict()))
    assert load_config(tmp_path / "config.json").retrieval == copied
    with mock.patch.object(storage, "configs_dir", return_value=tmp_path):
        path = storage.save_iteration_config(1, copied)
    assert json.loads(path.read_text()) == copied.to_dict()


@pytest.mark.parametrize("strategy", ["bm25", "hybrid"])
def test_missing_bm25_dependency_has_install_hint(strategy):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "rank_bm25":
            raise ModuleNotFoundError("No module named 'rank_bm25'")
        return original_import(name, *args, **kwargs)

    with mock.patch("builtins.__import__", side_effect=guarded_import):
        with pytest.raises(ImportError, match=r"pip install 'phoenix-rag\[retrieval\]'"):
            build_retriever(RetrievalConfig(retrieval_strategy=strategy), FakeStore())


def test_dense_pipeline_in_fresh_process_never_imports_rank_bm25():
    script = """
import builtins
import sys
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == 'rank_bm25' or name.startswith('rank_bm25.'):
        raise AssertionError('Dense must never import rank_bm25')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from unittest import mock
from phoenix_rag.config import RetrievalConfig
from phoenix_rag.core.rag_pipeline import RagPipeline
from phoenix_rag.retrieval.dense import DenseRetriever
from test_retrieval import FakeStore
generation = mock.Mock()
generation.chat.return_value = 'offline'
pipeline = RagPipeline(FakeStore(), generation, RetrievalConfig())
assert isinstance(pipeline._retriever, DenseRetriever)
assert pipeline.answer('q').answer == 'offline'
assert 'rank_bm25' not in sys.modules
"""
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True,
                   env={**os.environ, "PYTHONPATH": os.pathsep.join([str(root / "src"), str(root / "tests")])})
