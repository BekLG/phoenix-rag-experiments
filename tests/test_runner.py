"""Offline runs through real corpus registration, FAISS retrieval and persistence."""

from __future__ import annotations

import json
import unittest
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from phoenix_rag import storage, workspace
from phoenix_rag.benchmark.question_generator import load_benchmark, save_benchmark
from phoenix_rag.config import AppConfig, load_config, save_config
from phoenix_rag.core import corpus
from phoenix_rag.core.document_profile import compute_profile
from phoenix_rag.optimization import runner
from phoenix_rag.providers.base import ChatProvider, Providers

from test_corpus import FAISS_AVAILABLE, FakeEmbeddings, fake_questions, make_document


@unittest.skipUnless(FAISS_AVAILABLE, "faiss-cpu is not installed")
class CorpusRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = workspace.Workspace(temporary.name)
        self.enterContext(mock.patch.object(workspace, "_active", self.workspace))
        self.enterContext(mock.patch.object(storage, "_results_dir_override", None))
        self.source = make_document(self.workspace.root, "seed.txt")
        self.config = AppConfig()
        self.config.source_document = str(self.source)

        self.embeddings = FakeEmbeddings()
        generation = mock.Mock(spec=ChatProvider)
        generation.chat.return_value = "An answer based on the retrieved context."
        self.providers = Providers(
            embedding=self.embeddings, generation=generation,
            optimizer=mock.Mock(spec=ChatProvider), judge=mock.Mock(spec=ChatProvider),
        )
        self.enterContext(mock.patch.object(runner, "build_providers", return_value=self.providers))
        self.summarizer = self.enterContext(mock.patch.object(
            corpus, "generate_summary", return_value="Seed summary."
        ))
        self.generator = self.enterContext(mock.patch.object(
            corpus, "generate_benchmark", return_value=fake_questions("seed")
        ))
        self.evaluator = self.enterContext(mock.patch.object(
            runner, "run_evaluation", return_value={
                "faithfulness": 1.0, "context_recall": 1.0,
                "context_precision": 1.0, "response_relevancy": 1.0,
            }
        ))
        self.enterContext(mock.patch.object(
            runner, "propose_next_config_llm",
            side_effect=lambda current, *args: (current, []),
        ))

    def test_first_run_with_only_source_document_creates_a_working_corpus(self):
        self.assertIsNone(self.config.corpus_path)
        with mock.patch.object(
            corpus, "adopt_legacy_single_document_artifacts",
            wraps=corpus.adopt_legacy_single_document_artifacts,
        ) as adoption:
            result = runner.run_experiment(self.config)
            self.assertEqual(result["iteration"], 1)
            self.assertEqual(self.config.corpus_path, str(self.workspace.corpus_dir))
            state = corpus.load(self.config.corpus_path)
            self.assertEqual([d.label for d in state.documents], ["seed"])
            self.assertEqual(state.documents[0].path, str(self.source))
            self.assertEqual(corpus.optimization_status(state), "current")
            self.assertTrue(state.manifest_path.is_file())
            self.assertEqual(state.summary_text_path.read_text(), "Seed summary.")
            self.assertEqual(
                json.loads(state.profile_path.read_text()),
                asdict(compute_profile(self.source.read_text())),
            )
            self.assertEqual(load_benchmark(state.benchmark_path), fake_questions("seed"))
            self.assertEqual(len(state.indexes), 1)
            entry = next(iter(state.indexes.values()))
            self.assertEqual(entry.members, state.document_ids)
            self.assertEqual(entry.index_key, corpus.index_key(
                self.config.providers.embedding.model,
                result["config"].chunk_size, result["config"].chunk_overlap,
            ))
            self.assertTrue((state.index_path(entry.index_key) / "index.faiss").is_file())
            self.assertGreater(self.embeddings.call_count, 0)
            for answer in self.evaluator.call_args.args[0]:
                self.assertTrue(answer.contexts)
                self.assertTrue(all(s["doc_label"] == "seed" for s in answer.sources))

            # The next process loads the promoted config and reuses the corpus.
            saved = load_config(self.workspace.config_path)
            self.assertEqual(saved.corpus_path, self.config.corpus_path)
            self.embeddings.embedded_texts.clear()
            runner.run_experiment(saved)
            self.assertEqual(self.embeddings.call_count, 0)
            adoption.assert_called_once()
        self.summarizer.assert_called_once()
        self.generator.assert_called_once()
        self.assertFalse(Path(self.config.faiss_index_path).exists())
        self.assertFalse(Path(self.config.benchmark_path).exists())

    def test_upgrading_saved_config_adopts_caches_on_first_run(self):
        cached_profile = compute_profile(self.source.read_text())
        questions = fake_questions("legacy", 4)
        cache = self.workspace.generated_questions_dir
        cache.mkdir()
        Path(self.config.profile_path).write_text(json.dumps(asdict(cached_profile)))
        Path(self.config.summary_path).write_text("Paid-for legacy summary.")
        save_benchmark(questions, self.config.benchmark_path)
        original_caches = {p: p.read_bytes() for p in cache.iterdir()}
        save_config(self.config, self.workspace.config_path)
        upgraded = load_config(self.workspace.config_path)
        self.assertIsNone(upgraded.corpus_path)

        with mock.patch.object(
            corpus, "adopt_legacy_single_document_artifacts",
            wraps=corpus.adopt_legacy_single_document_artifacts,
        ) as adoption, mock.patch.object(corpus, "compute_profile") as profiler:
            result = runner.run_experiment(upgraded)

        adoption.assert_called_once()
        profiler.assert_not_called()
        self.summarizer.assert_not_called()
        self.generator.assert_not_called()
        self.assertEqual(result["iteration"], 1)
        state = corpus.load(upgraded.corpus_path)
        self.assertEqual(len(state.documents), 1)
        self.assertEqual(state.documents[0].document_profile, cached_profile)
        self.assertEqual(state.documents[0].summary, "Paid-for legacy summary.")
        self.assertEqual(state.documents[0].question_count, len(questions))
        self.assertEqual(load_benchmark(state.benchmark_path), questions)
        self.assertEqual(self.evaluator.call_args.args[1], questions)
        self.assertEqual({p: p.read_bytes() for p in cache.iterdir()}, original_caches)
        self.assertGreater(self.embeddings.call_count, 0)

    def test_explicit_corpus_location_is_preserved(self):
        self.config.corpus_path = str(self.workspace.root / "custom-corpus")
        runner.run_experiment(self.config)
        self.assertTrue((Path(self.config.corpus_path) / "manifest.json").is_file())
        self.assertFalse(self.workspace.corpus_dir.exists())

    def test_missing_seed_leaves_empty_corpus_and_reports_existing_corpus_error(self):
        self.source.unlink()
        with self.assertRaisesRegex(RuntimeError, "empty and could not be bootstrapped"):
            runner.run_experiment(self.config)
        self.assertTrue(corpus.load(self.config.corpus_path).is_empty)
        self.summarizer.assert_not_called()
        self.generator.assert_not_called()
        self.evaluator.assert_not_called()

    def test_existing_corpus_does_not_read_a_missing_seed_or_legacy_caches(self):
        runner.run_experiment(self.config)
        self.config.corpus_path = None  # An older saved config in the same workspace.
        self.config.source_document = str(self.workspace.root / "missing.pdf")
        self.summarizer.reset_mock()
        self.generator.reset_mock()
        self.embeddings.embedded_texts.clear()

        runner.run_experiment(self.config)

        self.summarizer.assert_not_called()
        self.generator.assert_not_called()
        self.assertEqual(self.embeddings.call_count, 0)
        self.assertEqual(len(corpus.load(self.config.corpus_path).documents), 1)
