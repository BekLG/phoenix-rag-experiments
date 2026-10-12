"""Offline chunk identity tests; no embeddings or model calls."""

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from phoenix_rag.core import corpus
from phoenix_rag.core.chunking import split_documents


def ids(chunks):
    return [chunk.metadata["chunk_id"] for chunk in chunks]


@pytest.mark.parametrize("size,overlap", [(40, 5), (70, 10)])
def test_split_text_and_metadata_unchanged_with_deterministic_unique_ids(tmp_path, size, overlap):
    text = "Repeated passage. " * 20
    source = tmp_path / "pages.txt"
    source.write_text(text * 2)
    pages = [
        Document(page_content=text, metadata={"source": str(source), "page": page, "label": "kept"})
        for page in range(2)
    ]
    original_metadata = [dict(page.metadata) for page in pages]
    old = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    ).split_documents(pages)

    chunks = split_documents(pages, size, overlap)

    assert [chunk.page_content for chunk in chunks] == [chunk.page_content for chunk in old]
    assert len(chunks) > 2
    assert len(set(ids(chunks))) == len(chunks)
    assert ids(chunks) == ids(split_documents(pages, size, overlap))
    assert {chunk.metadata["document_id"] for chunk in chunks} == {corpus.document_digest(source)}
    for chunk, previous in zip(chunks, old):
        assert {key: chunk.metadata[key] for key in previous.metadata} == previous.metadata
    assert [page.metadata for page in pages] == original_metadata


def test_file_content_identity_survives_rename_and_changes_with_content(tmp_path):
    source = tmp_path / "one.txt"
    source.write_text("A repeated sentence. " * 10)

    def chunks(path):
        return split_documents([Document(
            page_content=path.read_text(), metadata={"source": str(path)}
        )], 45, 5)

    original = chunks(source)
    renamed = tmp_path / "two.txt"
    source.rename(renamed)
    assert ids(original) == ids(chunks(renamed))
    renamed.write_text(renamed.read_text() + "Changed.")
    assert set(ids(original)).isdisjoint(ids(chunks(renamed)))


def test_manifest_identity_and_provenance_are_assigned_before_chunking(tmp_path):
    source = tmp_path / "corpus.txt"
    source.write_text("Text of a corpus document. " * 10)
    digest = corpus.document_digest(source)
    document = corpus.CorpusDocument(
        doc_id=digest, label="corpus-label", path=str(source), pages=None,
        characters=len(source.read_text()), added_at="", profile={}, summary="",
    )
    with mock.patch("phoenix_rag.core.chunk_identity.document_digest", side_effect=AssertionError(
        "The manifest digest should be reused without hashing the file again"
    )):
        chunks = corpus._chunks_for_document(document, 40, 5)
    assert chunks
    for chunk in chunks:
        assert chunk.metadata["document_id"] == chunk.metadata["doc_id"] == digest
        assert chunk.metadata["doc_label"] == "corpus-label"
        assert chunk.metadata["source"] == str(source)


def test_ids_ignore_other_documents_and_disambiguate_repeated_text():
    alpha = [Document(page_content="same", metadata={"doc_id": "alpha", "page": page})
             for page in range(3)]
    beta = Document(page_content="same", metadata={"doc_id": "beta"})
    alone = split_documents(alpha, 40, 5)
    together = split_documents([beta, *alpha], 40, 5)
    assert ids(alone) == ids(together)[1:]
    assert len(set(ids(together))) == 4


def test_document_identity_is_independent_of_chunk_parameters(tmp_path):
    source = tmp_path / "text.txt"
    source.write_text("Lots of text to split. " * 40)
    page = Document(page_content=source.read_text(), metadata={"source": str(source)})
    for size, overlap in [(40, 5), (90, 10)]:
        assert {c.metadata["document_id"] for c in split_documents([page], size, overlap)} == {
            corpus.document_digest(source)
        }


def test_in_memory_documents_and_empty_input():
    pages = [Document(page_content="in memory " * 20)]
    first = split_documents(pages, 40, 5)
    second = split_documents(pages, 80, 10)
    assert {c.metadata["document_id"] for c in first} == {c.metadata["document_id"] for c in second}
    assert len(set(ids(first))) == len(first)
    assert split_documents([], 40, 5) == []
    assert split_documents([Document(page_content="")], 40, 5) == []


def test_ids_match_across_processes_with_different_hash_seeds():
    script = """
import json
from langchain_core.documents import Document
from phoenix_rag.core.chunking import split_documents
chunks = split_documents([Document(page_content='Repeated text. ' * 20,
                                   metadata={'doc_id': 'source-digest'})], 40, 5)
print(json.dumps([c.metadata for c in chunks], sort_keys=True))
"""
    outputs = []
    for seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": seed,
               "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
        outputs.append(subprocess.check_output([sys.executable, "-c", script], env=env, text=True))
    assert json.loads(outputs[0]) == json.loads(outputs[1])
