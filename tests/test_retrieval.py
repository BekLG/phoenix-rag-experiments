"""Offline retrieval equivalence, pipeline and cache tests using fake embeddings."""

from dataclasses import FrozenInstanceError
from unittest import mock

import pytest
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from phoenix_rag.config import RetrievalConfig
from phoenix_rag.core.chunking import split_documents
from phoenix_rag.core.rag_pipeline import RagPipeline, RagResult
from phoenix_rag.core.vector_store import (
    add_chunks_to_store, load_vector_store, save_vector_store,
)
from phoenix_rag.evaluation.evaluator import build_ragas_dataset
from phoenix_rag.providers.base import ChatProvider
from phoenix_rag.retrieval.base import RetrievalResult
from phoenix_rag.retrieval.dense import DenseRetriever

from test_corpus import FAISS_AVAILABLE, FakeEmbeddings, fake_questions


METHODS = ("similarity", "mmr", "similarity_score_threshold")


def old_retriever(store, config):
    """Snapshot of RagPipeline's original construction, independent of new code."""
    if config.retriever_type == "similarity_score_threshold":
        return store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": config.top_k, "score_threshold": config.similarity_threshold},
        )
    if config.retriever_type == "mmr":
        return store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": config.top_k, "fetch_k": max(config.top_k * 4, 20)},
        )
    return store.as_retriever(search_type="similarity", search_kwargs={"k": config.top_k})


class FakeStore(VectorStore):
    """Known scores and tied hits, using the REAL LangChain retriever dispatcher."""

    def __init__(self):
        self.docs = [Document(page_content=text, metadata={"doc_id": "one", "page": i})
                     for i, text in enumerate(["  beta\n", "alpha", "gamma\n\ntext"])]
        self.index_to_docstore_id = {i: str(i) for i in range(len(self.docs))}
        self.docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(self.docs)})
        self.calls = []

    @classmethod
    def from_texts(cls, texts, embedding, metadatas=None, **kwargs):
        raise NotImplementedError

    def similarity_search_with_score(self, query, k=4, **kwargs):
        self.calls.append(("similarity", query, {"k": k, **kwargs}))
        return list(zip(self.docs, [0.1, 0.1, 0.7]))[:k]

    def similarity_search(self, query, k=4, **kwargs):
        return [doc for doc, _ in self.similarity_search_with_score(query, k, **kwargs)]

    def _select_relevance_score_fn(self):
        return lambda distance: 1.0 - distance

    def max_marginal_relevance_search(self, query, k=4, fetch_k=20, **kwargs):
        self.calls.append(("mmr", query, {"k": k, "fetch_k": fetch_k, **kwargs}))
        return [self.docs[2], self.docs[0], self.docs[1]][:k]


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("top_k", [1, 3, 6])
def test_fake_store_matches_old_text_order_parameters_and_scores(method, top_k):
    store = FakeStore()
    config = RetrievalConfig(retriever_type=method, top_k=top_k, similarity_threshold=0.5)
    old = old_retriever(store, config).invoke("query")
    previous_calls = list(store.calls)
    store.calls.clear()

    results = DenseRetriever(store, config).retrieve("query")

    assert [r.text for r in results] == [d.page_content for d in old]
    assert store.calls == previous_calls  # Same parameters, one search per query.
    assert [r.rank for r in results] == list(range(1, len(results) + 1))
    assert len({r.chunk_id for r in results}) == len(results)
    for result, document in zip(results, old):
        assert result.metadata == document.metadata
        assert result.metadata is not document.metadata
        assert result.document_id == "one"
        assert result.chunk_id == result.metadata["chunk_id"]
        assert result.retrieval_method == method
        assert result.reranker_score is None
        assert result.extras == {}
    expected = {"similarity": [0.1, 0.1, 0.7], "mmr": [None] * 3,
                "similarity_score_threshold": [0.9, 0.9]}
    assert [r.retrieval_score for r in results] == expected[method][:top_k]


@pytest.mark.parametrize("threshold,expected_count", [(0.9, 2), (0.95, 0)])
def test_threshold_boundary_and_empty_results_match_old_path(threshold, expected_count, caplog):
    store = FakeStore()
    config = RetrievalConfig(retriever_type="similarity_score_threshold", top_k=3,
                             similarity_threshold=threshold)
    old = old_retriever(store, config).invoke("query")
    with caplog.at_level("DEBUG", logger="phoenix_rag.retrieval.dense"):
        results = DenseRetriever(store, config).retrieve("query")
    assert len(results) == expected_count
    assert [r.text for r in results] == [doc.page_content for doc in old]
    assert f"method=similarity_score_threshold results={expected_count} latency_ms=" in caplog.text


def test_unknown_method_retains_pipeline_similarity_fallback():
    store = FakeStore()
    config = RetrievalConfig(retriever_type="legacy-unknown", top_k=2)
    results = DenseRetriever(store, config).retrieve("query")
    assert [r.text for r in results] == [d.page_content for d in old_retriever(store, config).invoke("query")]
    assert all(r.retrieval_method == "similarity" for r in results)


def test_result_is_frozen_and_extras_are_independent():
    first = RetrievalResult("chunk", "doc", "text", {}, 1, None, "mmr")
    second = RetrievalResult("chunk2", "doc", "text2", {}, 2, None, "mmr")
    with pytest.raises(FrozenInstanceError):
        first.rank = 3
    first.extras["future"] = 1
    assert second.extras == {}
    assert RagResult("question", "answer", ["context"]).retrieval_results is None


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("threshold", [0.5, 0.95])
def test_pipeline_prompt_and_ragas_contexts_match_old_path(method, threshold):
    store = FakeStore()
    config = RetrievalConfig(retriever_type=method, top_k=3, similarity_threshold=threshold)
    question = fake_questions("example", 1)[0]
    old = old_retriever(store, config).invoke(question.question)
    contexts = [doc.page_content for doc in old]
    generation = mock.Mock(spec=ChatProvider)
    generation.chat.return_value = "Generated answer"
    pipeline = RagPipeline(store, generation, config)

    result = pipeline.answer(question.question)

    assert result.contexts == contexts
    assert result.answer == "Generated answer"
    assert result.sources == [dict(doc.metadata) for doc in old]
    assert result.retrieval_results is not None
    assert [r.text for r in result.retrieval_results] == contexts
    assert [d.page_content for d in pipeline.retrieve(question.question)] == contexts
    generation.chat.assert_called_once_with(
        messages=[{"role": "user", "content": config.prompt_template.format(
            context="\n\n".join(contexts), question=question.question,
        )}], temperature=0.2,
    )
    previous = RagResult(question.question, result.answer, contexts)
    assert build_ragas_dataset([result], [question]).to_dict() == build_ragas_dataset(
        [previous], [question]
    ).to_dict()


@pytest.mark.skipif(not FAISS_AVAILABLE, reason="faiss-cpu is not installed")
@pytest.mark.parametrize("method", METHODS)
def test_real_faiss_with_fake_embeddings_matches_old_path(method):
    chunks = split_documents([Document(
        page_content="alpha\n\nbeta\n\ngamma\n\ndelta\n\nepsilon", metadata={"doc_id": "one"}
    )], 8, 0)
    store = FAISS.from_documents(chunks, FakeEmbeddings(),
                                 relevance_score_fn=lambda distance: 1 / (1 + distance))
    config = RetrievalConfig(retriever_type=method, top_k=3, similarity_threshold=0.8)
    old = old_retriever(store, config).invoke("alpha")
    results = DenseRetriever(store, config).retrieve("alpha")
    assert results
    assert [r.text for r in results] == [doc.page_content for doc in old]
    assert [r.metadata for r in results] == [doc.metadata for doc in old]
    if method != "mmr":
        assert results[0].retrieval_score == (0.0 if method == "similarity" else 1.0)


def stored_documents(store):
    return [store.docstore.search(key) for _, key in sorted(store.index_to_docstore_id.items())]


@pytest.mark.skipif(not FAISS_AVAILABLE, reason="faiss-cpu is not installed")
@pytest.mark.parametrize("has_manifest_digest", [True, False])
def test_legacy_cache_backfill_matches_fresh_ids_without_embedding_or_rewriting(tmp_path, has_manifest_digest):
    source = tmp_path / "source.txt"
    source.write_text("repeated\n\nrepeated\n\nlast")
    metadata = {"source": str(source), "doc_label": "label"}
    if has_manifest_digest:
        from phoenix_rag.core.corpus import document_digest
        metadata["doc_id"] = document_digest(source)
    fresh = split_documents([Document(page_content=source.read_text(), metadata=metadata)], 12, 0)
    legacy = [Document(page_content=c.page_content, metadata={
        k: v for k, v in c.metadata.items() if k not in ("chunk_id", "document_id")
    }) for c in fresh]
    embeddings = FakeEmbeddings()
    store = FAISS.from_documents(legacy, embeddings)
    cache = tmp_path / "index"
    save_vector_store(store, cache)
    before = {path.name: path.read_bytes() for path in cache.iterdir()}
    if has_manifest_digest:
        source.unlink()  # Corpus caches retain identity even without the source.
    embeddings.embedded_texts.clear()

    loaded = load_vector_store(cache, embeddings)

    assert embeddings.call_count == 0
    assert loaded.index.ntotal == store.index.ntotal
    assert loaded.index_to_docstore_id == store.index_to_docstore_id
    assert [d.metadata for d in stored_documents(loaded)] == [c.metadata for c in fresh]
    assert {path.name: path.read_bytes() for path in cache.iterdir()} == before
    for query in ("last", "repeated"):
        old = old_retriever(store, RetrievalConfig(top_k=len(fresh))).invoke(query)
        results = DenseRetriever(loaded, RetrievalConfig(top_k=len(fresh))).retrieve(query)
        assert [r.text for r in results] == [d.page_content for d in old]
        assert len({r.chunk_id for r in results}) == len(results)
        assert {r.chunk_id for r in results} == {c.metadata["chunk_id"] for c in fresh}
    # A separate load derives the same IDs without persisting a migration.
    assert [d.metadata for d in stored_documents(load_vector_store(cache, embeddings))] == [
        c.metadata for c in fresh
    ]


@pytest.mark.skipif(not FAISS_AVAILABLE, reason="faiss-cpu is not installed")
def test_directly_loaded_and_mixed_legacy_cache_keeps_ids_when_extended(tmp_path):
    fresh = split_documents([
        Document(page_content="same", metadata={"doc_id": "old"}),
        Document(page_content="same", metadata={"doc_id": "old"}),
    ], 40, 0)
    mixed = [fresh[0], Document(page_content=fresh[1].page_content, metadata={"doc_id": "old"})]
    embeddings = FakeEmbeddings()
    store = FAISS.from_documents(mixed, embeddings)
    save_vector_store(store, tmp_path)
    # Bypass our loader, as an existing caller may do.
    loaded = FAISS.load_local(str(tmp_path), embeddings, allow_dangerous_deserialization=True)
    DenseRetriever(loaded, RetrievalConfig())
    assert [d.metadata for d in stored_documents(loaded)] == [c.metadata for c in fresh]

    new = split_documents([Document(page_content="same", metadata={"doc_id": "new"})], 4, 0)
    embeddings.embedded_texts.clear()
    add_chunks_to_store(loaded, new)
    save_vector_store(loaded, tmp_path)
    reloaded = load_vector_store(tmp_path, embeddings)
    assert embeddings.call_count == len(new)
    assert [d.metadata for d in stored_documents(reloaded)] == [c.metadata for c in [*fresh, *new]]


@pytest.mark.skipif(not FAISS_AVAILABLE, reason="faiss-cpu is not installed")
def test_legacy_cache_without_source_or_digest_uses_stable_fallback(tmp_path):
    docs = [Document(page_content=text, metadata={"source": str(tmp_path / "missing.txt")})
            for text in ["repeat", "repeat", "last"]]
    embeddings = FakeEmbeddings()
    save_vector_store(FAISS.from_documents(docs, embeddings), tmp_path / "index")
    first = stored_documents(load_vector_store(tmp_path / "index", embeddings))
    second = stored_documents(load_vector_store(tmp_path / "index", embeddings))
    assert [d.metadata for d in first] == [d.metadata for d in second]
    assert len({d.metadata["document_id"] for d in first}) == 1
    assert len({d.metadata["chunk_id"] for d in first}) == 3
