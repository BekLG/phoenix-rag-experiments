"""
phoenix_rag.core
================
The RAG pipeline itself and the pieces that describe what is being indexed:
document loading, chunking, embeddings, the FAISS store, the retrieve-and-answer
pipeline, document profiling, seed-config derivation, and the multi-document
corpus.

Deliberately no re-exports: importing any one of these pulls in langchain and
faiss, and a caller that wants ``split_documents`` should not pay for the vector
store. Import from the specific module.
"""
