"""
test_corpus.py
==============
Offline tests for the multi-document corpus.

Nothing here touches the Mistral API. Summary and question generation are patched
out, and embeddings come from FakeEmbeddings below, which also COUNTS how many
texts it was asked to embed -- that counter is the actual assertion behind the
central claim of this feature: adding a document to an existing index embeds only
the new document's chunks.
"""

from __future__ import annotations

import hashlib
import json
import unittest
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from langchain_core.embeddings import Embeddings

import corpus
from config import AppConfig
from document_profile import DocumentProfile, aggregate_profiles, compute_profile
from question_generator import BenchmarkQuestion, save_benchmark

try:  # faiss is a heavy optional dependency; the manifest tests do not need it.
    import faiss  # noqa: F401

    FAISS_AVAILABLE = True
except ImportError:  # pragma: no cover
    FAISS_AVAILABLE = False


class FakeEmbeddings(Embeddings):
    """Deterministic embeddings that record how much work they were asked to do.

    A hash-derived vector rather than a constant one: FAISS over identical vectors
    makes every similarity assertion degenerate, and a real bug in chunk
    attribution would hide behind it. hashlib rather than hash() because str
    hashing is salted per process, and these vectors get written to disk and read
    back within a test.
    """

    def __init__(self, dimension: int = 16):
        self.dimension = dimension
        self.embedded_texts: list[str] = []

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [digest[i % len(digest)] / 255.0 for i in range(self.dimension)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embedded_texts.extend(texts)
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    @property
    def call_count(self) -> int:
        return len(self.embedded_texts)


def make_document(directory: Path, name: str, paragraphs: int = 6) -> Path:
    """A .txt document long enough to split into several chunks."""
    path = directory / name
    body = "\n\n".join(
        f"Section {index} of {name}. " + f"Content line about {name} topic {index}. " * 8
        for index in range(paragraphs)
    )
    path.write_text(body, encoding="utf-8")
    return path


def fake_questions(label: str, count: int = 2) -> list[BenchmarkQuestion]:
    return [
        BenchmarkQuestion(
            question=f"What does {label} say about topic {index}?",
            reference_answer=f"It discusses topic {index}.",
            reference_context=f"Section {index} of {label}.",
            question_type="factual",
        )
        for index in range(count)
    ]


class CorpusTestCase(unittest.TestCase):
    """Shared fixture: a temp corpus root and an AppConfig pointing into it."""

    def setUp(self) -> None:
        self._temp = TemporaryDirectory()
        self.root = Path(self._temp.name)
        self.docs_dir = self.root / "docs"
        self.docs_dir.mkdir()
        self.corpus_root = self.root / "corpus"

        self.app_config = AppConfig()
        self.app_config.mistral.api_key = "test-key-not-used"
        self.app_config.corpus_path = str(self.corpus_root)
        self.app_config.source_document = str(self.root / "nonexistent.pdf")
        self.app_config.benchmark_path = str(self.root / "benchmark.json")
        self.app_config.summary_path = str(self.root / "summary.txt")
        self.app_config.profile_path = str(self.root / "profile.json")

    def tearDown(self) -> None:
        self._temp.cleanup()

    def register(self, state, path: Path, label: str, questions: int = 2):
        """register_document with the two API-calling helpers patched out."""
        with mock.patch.object(
            corpus, "generate_summary", return_value=f"Summary of {label}."
        ), mock.patch.object(
            corpus, "generate_benchmark", return_value=fake_questions(label, questions)
        ):
            return corpus.register_document(
                state, self.app_config, path=path, label=label
            )


# =====================================================================
# Manifest
# =====================================================================

class ManifestTests(CorpusTestCase):
    def test_load_of_missing_root_gives_an_empty_corpus(self):
        state = corpus.load(self.corpus_root)
        self.assertTrue(state.is_empty)
        self.assertEqual(state.documents, [])
        self.assertEqual(state.indexes, {})

    def test_round_trip_preserves_documents_and_indexes(self):
        state = corpus.load(self.corpus_root)
        path = make_document(self.docs_dir, "alpha.txt")
        document = self.register(state, path, "alpha")
        self.assertIsNotNone(document)

        state.indexes["k1"] = corpus.CorpusIndex(
            index_key="k1",
            embedding_model="mistral-embed",
            chunk_size=800,
            chunk_overlap=120,
            members=[document.doc_id],
        )
        corpus.save(state)

        reloaded = corpus.load(self.corpus_root)
        self.assertEqual(len(reloaded.documents), 1)
        self.assertEqual(reloaded.documents[0].label, "alpha")
        self.assertEqual(reloaded.documents[0].doc_id, document.doc_id)
        self.assertEqual(reloaded.documents[0].summary, "Summary of alpha.")
        self.assertEqual(reloaded.indexes["k1"].chunk_size, 800)
        self.assertEqual(reloaded.indexes["k1"].members, [document.doc_id])
        # The profile survives as a DocumentProfile, not just a dict -- seed_config
        # and the optimizer both need the dataclass.
        self.assertIsInstance(reloaded.documents[0].document_profile, DocumentProfile)

    def test_manifest_is_valid_json_with_a_version(self):
        state = corpus.load(self.corpus_root)
        self.register(state, make_document(self.docs_dir, "alpha.txt"), "alpha")
        data = json.loads(state.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(data["version"], corpus.MANIFEST_VERSION)
        self.assertIn("documents", data)
        self.assertIn("indexes", data)

    def test_readding_identical_content_is_a_noop(self):
        state = corpus.load(self.corpus_root)
        path = make_document(self.docs_dir, "alpha.txt")
        first = self.register(state, path, "alpha")
        self.assertIsNotNone(first)

        # Same bytes under a different filename: still the same document, because
        # identity is the content digest, not the path.
        copy_path = self.docs_dir / "alpha_copy.txt"
        copy_path.write_bytes(path.read_bytes())
        again = self.register(state, copy_path, "alpha-copy")

        self.assertIsNone(again)
        self.assertEqual(len(state.documents), 1)

    def test_duplicate_labels_are_suffixed_not_rejected(self):
        state = corpus.load(self.corpus_root)
        first = self.register(state, make_document(self.docs_dir, "a.txt"), "notes")
        second = self.register(
            state, make_document(self.docs_dir, "b.txt", paragraphs=4), "notes"
        )
        self.assertEqual(first.label, "notes")
        self.assertNotEqual(second.label, "notes")
        self.assertTrue(second.label.startswith("notes"))

    def test_missing_file_is_reported_by_verify_documents(self):
        state = corpus.load(self.corpus_root)
        path = make_document(self.docs_dir, "alpha.txt")
        self.register(state, path, "alpha")
        self.assertEqual(corpus.verify_documents(state), [])

        path.unlink()
        problems = corpus.verify_documents(state)
        self.assertEqual(len(problems), 1)
        self.assertIn("alpha", problems[0])

    def test_edited_file_is_reported_by_verify_documents(self):
        state = corpus.load(self.corpus_root)
        path = make_document(self.docs_dir, "alpha.txt")
        self.register(state, path, "alpha")

        path.write_text("completely different content", encoding="utf-8")
        problems = corpus.verify_documents(state)
        self.assertEqual(len(problems), 1)
        self.assertIn("alpha", problems[0])


# =====================================================================
# Benchmark growth
# =====================================================================

class BenchmarkGrowthTests(CorpusTestCase):
    def test_benchmark_accumulates_across_documents(self):
        state = corpus.load(self.corpus_root)
        first = self.register(
            state, make_document(self.docs_dir, "alpha.txt"), "alpha", questions=3
        )
        self.assertEqual(first.question_count, 3)

        second = self.register(
            state, make_document(self.docs_dir, "beta.txt"), "beta", questions=2
        )
        self.assertEqual(second.question_count, 2)

        questions = corpus.corpus_benchmark(state, self.app_config)
        self.assertEqual(len(questions), 5)
        # The older document's questions are still there -- growing the corpus must
        # not stop testing what was already in it.
        self.assertTrue(any("alpha" in q.question for q in questions))
        self.assertTrue(any("beta" in q.question for q in questions))

    def test_duplicate_questions_are_deduped_and_not_counted(self):
        state = corpus.load(self.corpus_root)
        self.register(state, make_document(self.docs_dir, "alpha.txt"), "alpha", questions=2)

        # A second document that happens to generate the SAME questions.
        with mock.patch.object(corpus, "generate_summary", return_value="Summary."), \
             mock.patch.object(
                 corpus, "generate_benchmark", return_value=fake_questions("alpha", 2)
             ):
            second = corpus.register_document(
                state,
                self.app_config,
                path=make_document(self.docs_dir, "beta.txt"),
                label="beta",
            )

        self.assertEqual(second.question_count, 0)
        self.assertEqual(len(corpus.corpus_benchmark(state, self.app_config)), 2)

    def test_registering_without_question_generation_leaves_the_benchmark_alone(self):
        state = corpus.load(self.corpus_root)
        self.register(state, make_document(self.docs_dir, "alpha.txt"), "alpha", questions=2)

        with mock.patch.object(corpus, "generate_summary", return_value="Summary."), \
             mock.patch.object(corpus, "generate_benchmark") as generator:
            document = corpus.register_document(
                state,
                self.app_config,
                path=make_document(self.docs_dir, "beta.txt"),
                label="beta",
                generate_questions=False,
            )

        generator.assert_not_called()
        self.assertEqual(document.question_count, 0)
        self.assertEqual(len(corpus.corpus_benchmark(state, self.app_config)), 2)


# =====================================================================
# Incremental index maintenance -- the point of the feature
# =====================================================================

@unittest.skipUnless(FAISS_AVAILABLE, "faiss-cpu is not installed")
class SyncIndexTests(CorpusTestCase):
    CHUNK_SIZE = 400
    CHUNK_OVERLAP = 40

    def sync(self, state, embeddings, chunk_size=None, chunk_overlap=None):
        return corpus.sync_index(
            state,
            embeddings=embeddings,
            embedding_model="fake-embed",
            chunk_size=chunk_size or self.CHUNK_SIZE,
            chunk_overlap=self.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap,
        )

    def test_empty_corpus_refuses_to_build(self):
        state = corpus.load(self.corpus_root)
        with self.assertRaises(ValueError):
            self.sync(state, FakeEmbeddings())

    def test_first_sync_builds_over_every_member(self):
        state = corpus.load(self.corpus_root)
        self.register(state, make_document(self.docs_dir, "alpha.txt"), "alpha")
        self.register(state, make_document(self.docs_dir, "beta.txt"), "beta")

        embeddings = FakeEmbeddings()
        store, report = self.sync(state, embeddings)

        self.assertEqual(report.action, "built")
        self.assertEqual(sorted(report.added_labels), ["alpha", "beta"])
        self.assertGreater(report.vectors_after, 0)
        self.assertEqual(report.vectors_after, corpus.vector_count(store))
        self.assertEqual(embeddings.call_count, report.added_chunks)
        self.assertTrue((report.index_path / corpus.COMPLETE_MARKER).exists())

        # Membership is recorded per variant, which is what lets a later add
        # be recognised as incremental.
        entry = state.indexes[report.index_key]
        self.assertEqual(sorted(entry.members), sorted(state.document_ids))

    def test_second_sync_with_no_new_documents_reuses_the_index(self):
        state = corpus.load(self.corpus_root)
        self.register(state, make_document(self.docs_dir, "alpha.txt"), "alpha")
        self.sync(state, FakeEmbeddings())

        reloaded = corpus.load(self.corpus_root)
        embeddings = FakeEmbeddings()
        _store, report = self.sync(reloaded, embeddings)

        self.assertEqual(report.action, "loaded")
        self.assertEqual(report.added_chunks, 0)
        self.assertEqual(report.vectors_before, report.vectors_after)
        # Nothing re-embedded. A single embed_documents call here would mean the
        # index was silently rebuilt.
        self.assertEqual(embeddings.call_count, 0)

    def test_adding_a_document_embeds_only_the_new_chunks(self):
        """The central claim: an add extends the index instead of rebuilding it."""
        state = corpus.load(self.corpus_root)
        self.register(state, make_document(self.docs_dir, "alpha.txt"), "alpha")
        self.register(state, make_document(self.docs_dir, "beta.txt"), "beta")

        build_embeddings = FakeEmbeddings()
        _store, build_report = self.sync(state, build_embeddings)
        chunks_for_two = build_report.added_chunks
        index_path = build_report.index_path

        # A third document arrives after the index already exists.
        reloaded = corpus.load(self.corpus_root)
        gamma_path = make_document(self.docs_dir, "gamma.txt", paragraphs=4)
        self.register(reloaded, gamma_path, "gamma")

        add_embeddings = FakeEmbeddings()
        store, report = self.sync(reloaded, add_embeddings)

        self.assertEqual(report.action, "extended")
        self.assertEqual(report.added_labels, ["gamma"])
        self.assertGreater(report.added_chunks, 0)
        self.assertLess(report.added_chunks, chunks_for_two)

        # Only gamma's chunks were sent to the embedder.
        self.assertEqual(add_embeddings.call_count, report.added_chunks)
        self.assertTrue(
            all("gamma" in text for text in add_embeddings.embedded_texts),
            "chunks from an already-indexed document were re-embedded",
        )

        # The index grew in place: same directory, more vectors.
        self.assertEqual(report.index_path, index_path)
        self.assertEqual(report.vectors_before, chunks_for_two)
        self.assertEqual(report.vectors_after, chunks_for_two + report.added_chunks)
        self.assertEqual(corpus.vector_count(store), report.vectors_after)

        # And all three documents are retrievable from the one shared index, with
        # provenance metadata intact.
        #
        # Each probe is the exact text of a chunk that is already stored. embed_query
        # is deterministic and shares _vector with embed_documents, so the probe's
        # vector IS that chunk's stored vector -- L2 distance 0, necessarily the
        # nearest hit. A semantic query like "gamma topic" would prove nothing here:
        # hash-derived vectors carry no relationship between a query and the document
        # it came from, so its top-k is arbitrary and gamma (4 paragraphs against
        # alpha and beta's 6 each) loses that lottery on chunk count alone.
        probes = {
            "gamma": add_embeddings.embedded_texts[0],
            "alpha": next(t for t in build_embeddings.embedded_texts if "alpha" in t),
            "beta": next(t for t in build_embeddings.embedded_texts if "beta" in t),
        }
        for label, probe in probes.items():
            with self.subTest(document=label):
                hit = store.similarity_search(probe, k=1)[0]
                self.assertEqual(hit.metadata.get("doc_label"), label)

    def test_changing_chunk_size_builds_a_separate_variant(self):
        state = corpus.load(self.corpus_root)
        self.register(state, make_document(self.docs_dir, "alpha.txt"), "alpha")

        _store, first = self.sync(state, FakeEmbeddings())

        embeddings = FakeEmbeddings()
        _store, second = self.sync(state, embeddings, chunk_size=900, chunk_overlap=90)

        # Differently-sized chunks are different vectors, so this must be a full
        # build into its own directory rather than an extension of the first.
        self.assertEqual(second.action, "built")
        self.assertNotEqual(second.index_key, first.index_key)
        self.assertNotEqual(second.index_path, first.index_path)
        self.assertEqual(embeddings.call_count, second.added_chunks)
        self.assertEqual(len(state.indexes), 2)
        # The original variant survives, so switching back is free.
        self.assertTrue((first.index_path / corpus.COMPLETE_MARKER).exists())

    def test_document_added_under_one_variant_is_picked_up_by_another(self):
        state = corpus.load(self.corpus_root)
        self.register(state, make_document(self.docs_dir, "alpha.txt"), "alpha")
        self.sync(state, FakeEmbeddings())                       # variant A
        self.sync(state, FakeEmbeddings(), chunk_size=900)       # variant B

        self.register(state, make_document(self.docs_dir, "beta.txt"), "beta")

        _store, report_a = self.sync(state, FakeEmbeddings())
        _store, report_b = self.sync(state, FakeEmbeddings(), chunk_size=900)

        self.assertEqual(report_a.action, "extended")
        self.assertEqual(report_b.action, "extended")
        self.assertEqual(report_a.added_labels, ["beta"])
        self.assertEqual(report_b.added_labels, ["beta"])

    def test_a_corrupt_index_is_rebuilt_rather_than_crashing(self):
        state = corpus.load(self.corpus_root)
        self.register(state, make_document(self.docs_dir, "alpha.txt"), "alpha")
        _store, first = self.sync(state, FakeEmbeddings())

        (first.index_path / "index.faiss").write_bytes(b"not a faiss index")

        embeddings = FakeEmbeddings()
        _store, report = self.sync(state, embeddings)
        self.assertEqual(report.action, "built")
        self.assertGreater(embeddings.call_count, 0)

    def test_removing_a_document_invalidates_the_variants_that_held_it(self):
        state = corpus.load(self.corpus_root)
        alpha = self.register(state, make_document(self.docs_dir, "alpha.txt"), "alpha")
        self.register(state, make_document(self.docs_dir, "beta.txt"), "beta")
        _store, built = self.sync(state, FakeEmbeddings())

        removed = corpus.remove_document(state, alpha.doc_id)
        self.assertIsNotNone(removed)
        self.assertEqual(len(state.documents), 1)

        # FAISS has no cheap per-vector removal, so the variant is invalidated and
        # the next sync rebuilds it from the remaining documents.
        self.assertFalse((built.index_path / corpus.COMPLETE_MARKER).exists())
        embeddings = FakeEmbeddings()
        _store, report = self.sync(state, embeddings)
        self.assertEqual(report.action, "built")
        self.assertEqual(report.added_labels, ["beta"])
        self.assertTrue(all("alpha" not in text for text in embeddings.embedded_texts))


# =====================================================================
# Aggregated artifacts the optimizer sees
# =====================================================================

class AggregatedArtifactTests(CorpusTestCase):
    def test_single_document_summary_is_passed_through_verbatim(self):
        state = corpus.load(self.corpus_root)
        self.register(state, make_document(self.docs_dir, "alpha.txt"), "alpha")
        # A one-document corpus must look exactly like single-document mode to the
        # optimizer, or enabling corpus mode would change its behaviour by itself.
        self.assertEqual(corpus.summary_text(state), "Summary of alpha.")

    def test_multi_document_summary_names_every_document(self):
        state = corpus.load(self.corpus_root)
        self.register(state, make_document(self.docs_dir, "alpha.txt"), "alpha")
        self.register(state, make_document(self.docs_dir, "beta.txt"), "beta")

        text = corpus.summary_text(state)
        self.assertIn("Summary of alpha.", text)
        self.assertIn("Summary of beta.", text)
        self.assertIn("alpha", text)
        self.assertIn("beta", text)
        # The header has to tell the optimizer not to scope its prompt template to
        # one subject, which is the failure mode of feeding it a merged summary.
        self.assertIn("2-DOCUMENT", text.upper())

    def test_profile_aggregates_across_documents(self):
        state = corpus.load(self.corpus_root)
        first = self.register(state, make_document(self.docs_dir, "alpha.txt"), "alpha")
        second = self.register(
            state, make_document(self.docs_dir, "beta.txt", paragraphs=3), "beta"
        )

        aggregated = corpus.profile(state)
        self.assertEqual(
            aggregated.characters,
            first.document_profile.characters + second.document_profile.characters,
        )

    def test_rendered_artifacts_stay_inside_the_corpus_root(self):
        state = corpus.load(self.corpus_root)
        self.register(state, make_document(self.docs_dir, "alpha.txt"), "alpha")

        self.assertTrue(state.summary_text_path.exists())
        self.assertTrue(state.profile_path.exists())
        # The single-document caches must be untouched, or a later single-document
        # run would silently inherit corpus artifacts.
        self.assertFalse(Path(self.app_config.summary_path).exists())
        self.assertFalse(Path(self.app_config.profile_path).exists())
        self.assertFalse(Path(self.app_config.benchmark_path).exists())


class OptimizationStatusTests(CorpusTestCase):
    def test_status_moves_never_to_current_to_stale(self):
        state = corpus.load(self.corpus_root)
        self.register(state, make_document(self.docs_dir, "alpha.txt"), "alpha")
        self.assertEqual(corpus.optimization_status(state), "never")

        corpus.mark_optimized(state)
        self.assertEqual(corpus.optimization_status(state), "current")

        self.register(state, make_document(self.docs_dir, "beta.txt"), "beta")
        self.assertEqual(corpus.optimization_status(state), "stale")

        corpus.mark_optimized(state)
        self.assertEqual(corpus.optimization_status(state), "current")


class BootstrapTests(CorpusTestCase):
    def test_bootstrap_adopts_cached_artifacts_without_generating(self):
        source = make_document(self.docs_dir, "existing.txt")
        self.app_config.source_document = str(source)
        # The realistic starting point: a previous single-document run already paid
        # for all three artifacts.
        Path(self.app_config.summary_path).write_text(
            "Cached summary from an earlier run.", encoding="utf-8"
        )
        Path(self.app_config.profile_path).write_text(
            json.dumps(asdict(compute_profile(source.read_text(encoding="utf-8")))),
            encoding="utf-8",
        )
        save_benchmark(fake_questions("existing", 4), self.app_config.benchmark_path)

        state = corpus.load(self.corpus_root)
        with mock.patch.object(corpus, "generate_summary") as summarizer, \
             mock.patch.object(corpus, "generate_benchmark") as generator:
            document = corpus.bootstrap_from_single_document(state, self.app_config)

        self.assertIsNotNone(document)
        # The whole point: converting a single-document setup into a corpus spends
        # nothing, because these artifacts were already paid for.
        summarizer.assert_not_called()
        generator.assert_not_called()
        self.assertEqual(document.summary, "Cached summary from an earlier run.")
        self.assertEqual(document.question_count, 4)
        self.assertEqual(len(corpus.corpus_benchmark(state, self.app_config)), 4)

    def test_bootstrap_is_a_noop_on_a_populated_corpus(self):
        state = corpus.load(self.corpus_root)
        self.register(state, make_document(self.docs_dir, "alpha.txt"), "alpha")
        self.app_config.source_document = str(make_document(self.docs_dir, "other.txt"))

        self.assertIsNone(corpus.bootstrap_from_single_document(state, self.app_config))
        self.assertEqual(len(state.documents), 1)

    def test_bootstrap_of_a_missing_source_document_is_survivable(self):
        state = corpus.load(self.corpus_root)
        self.app_config.source_document = str(self.root / "does_not_exist.pdf")
        self.assertIsNone(corpus.bootstrap_from_single_document(state, self.app_config))
        self.assertTrue(state.is_empty)


class IndexKeyTests(unittest.TestCase):
    def test_key_depends_only_on_the_embedding_variant(self):
        base = corpus.index_key("mistral-embed", 800, 120)
        self.assertEqual(base, corpus.index_key("mistral-embed", 800, 120))
        self.assertNotEqual(base, corpus.index_key("mistral-embed", 801, 120))
        self.assertNotEqual(base, corpus.index_key("mistral-embed", 800, 121))
        self.assertNotEqual(base, corpus.index_key("other-embed", 800, 120))


class AggregateProfilesTests(unittest.TestCase):
    @staticmethod
    def profile(**overrides) -> DocumentProfile:
        defaults = dict(
            pages=10,
            characters=1000,
            estimated_tokens=250,
            sections=5,
            median_chars_per_page=100.0,
            min_chars_per_page=50.0,
            max_chars_per_page=150.0,
            doc_type="report",
            table_heavy=False,
            list_heavy=False,
        )
        defaults.update(overrides)
        return DocumentProfile(**defaults)

    def test_empty_list_returns_an_empty_profile(self):
        aggregated = aggregate_profiles([])
        self.assertEqual(aggregated.characters, 0)

    def test_single_profile_is_returned_unchanged(self):
        one = self.profile()
        self.assertIs(aggregate_profiles([one]), one)

    def test_additive_fields_are_summed(self):
        aggregated = aggregate_profiles([
            self.profile(pages=10, characters=1000, estimated_tokens=250, sections=5),
            self.profile(pages=4, characters=400, estimated_tokens=100, sections=2),
        ])
        self.assertEqual(aggregated.pages, 14)
        self.assertEqual(aggregated.characters, 1400)
        self.assertEqual(aggregated.estimated_tokens, 350)
        self.assertEqual(aggregated.sections, 7)

    def test_extremes_span_every_document(self):
        aggregated = aggregate_profiles([
            self.profile(min_chars_per_page=50.0, max_chars_per_page=150.0),
            self.profile(min_chars_per_page=20.0, max_chars_per_page=900.0),
        ])
        self.assertEqual(aggregated.min_chars_per_page, 20.0)
        self.assertEqual(aggregated.max_chars_per_page, 900.0)

    def test_median_is_page_weighted(self):
        # 1 page at 1000 chars/page vs 9 pages at 100: the median must follow the
        # bulk of the pages, not average the two documents.
        aggregated = aggregate_profiles([
            self.profile(pages=1, median_chars_per_page=1000.0),
            self.profile(pages=9, median_chars_per_page=100.0),
        ])
        self.assertEqual(aggregated.median_chars_per_page, 100.0)

    def test_doc_type_is_character_weighted_and_never_invented(self):
        aggregated = aggregate_profiles([
            self.profile(characters=100, doc_type="slides"),
            self.profile(characters=9000, doc_type="paper"),
        ])
        # Must be one of the inputs: seed_config._select_regime and the optimizer's
        # sizing policy branch on compute_profile's exact vocabulary, so a new
        # label like "mixed" would silently drop into their fallback paths.
        self.assertEqual(aggregated.doc_type, "paper")

    def test_none_page_counts_do_not_poison_the_sum(self):
        aggregated = aggregate_profiles([
            self.profile(pages=None),
            self.profile(pages=6),
        ])
        self.assertEqual(aggregated.pages, 6)

        both_none = aggregate_profiles([
            self.profile(pages=None), self.profile(pages=None, characters=1)
        ])
        self.assertIsNone(both_none.pages)

    def test_heavy_flags_need_half_the_characters(self):
        minority = aggregate_profiles([
            self.profile(characters=100, table_heavy=True),
            self.profile(characters=900, table_heavy=False),
        ])
        self.assertFalse(minority.table_heavy)

        majority = aggregate_profiles([
            self.profile(characters=900, list_heavy=True),
            self.profile(characters=100, list_heavy=False),
        ])
        self.assertTrue(majority.list_heavy)


if __name__ == "__main__":
    unittest.main()
