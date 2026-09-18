"""
phoenix_rag.benchmark
=====================
Building the fixed evaluation set: question generation and document
summarization.

The benchmark is generated once from the COMPLETE source document and then held
constant for the whole optimization run, so every configuration is scored
against identical questions. See
:mod:`~phoenix_rag.benchmark.question_generator` for why it is never generated
from retrieved chunks.
"""
