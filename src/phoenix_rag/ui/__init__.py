"""
phoenix_rag.ui
==============
Operator front-ends. Both are thin presentation layers over
:mod:`phoenix_rag.operations`, which owns every decision they appear to make, so
the two cannot drift apart in behaviour.

    python -m phoenix_rag.ui.menu
    streamlit run src/phoenix_rag/ui/streamlit_app.py

The GUI needs the optional extra: ``pip install phoenix-rag[ui]``.
"""
