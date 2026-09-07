"""
phoenix_rag.providers
=====================
Backend clients: the things that actually call an embedding or chat API.

``mistral`` is the only implementation so far. The shared machinery that is not
specific to any one backend -- request pacing and retry -- lives in
:mod:`~phoenix_rag.providers.ratelimit` so the next backend inherits it rather
than reimplementing it.
"""
