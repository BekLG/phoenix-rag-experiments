"""
phoenix_rag.optimization
========================
The optimization loop: the LLM-driven proposer that writes the next retrieval
configuration and answer prompt, the bounds/target helpers it is clamped by, and
the runner that orchestrates index build -> answer -> score -> propose.
"""
