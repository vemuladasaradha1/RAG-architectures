# RAG Architectures

A local-first, open-source learning and production reference for 16 Retrieval-Augmented Generation architectures. Every architecture has both a reusable Python implementation and a dedicated Jupyter notebook.

## Architectures

1. Basic RAG
2. Dense / semantic RAG
3. Sparse / BM25 RAG
4. Hybrid RAG
5. Reranking RAG
6. Multi-query RAG
7. HyDE
8. Contextual RAG
9. Parent-child RAG
10. Hierarchical RAG
11. Multi-hop RAG
12. GraphRAG
13. Corrective RAG
14. Self-RAG
15. Adaptive RAG
16. Agentic RAG

## Local-first and API-key free

The core examples use a dependency-light in-memory backend and a deterministic extractive generator, so **no API key is required**. For realistic experiments, the notebooks show optional upgrades to local open-weight Transformers models, Qdrant/FAISS, BM25, rerankers, LangGraph, and GraphRAG. Public model downloads may require internet access, but not a paid API key.

## Notebooks

Each notebook is deliberately modular: run the cells top-to-bottom, or replace individual components (loader, chunker, retriever, reranker, generator, evaluator) independently. The notebook uses the same package modules as the application to prevent notebook-only code drift.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -e '.[notebooks,dev]'
pytest
jupyter lab
```

## Production direction

Use Qdrant/FAISS for vector search, BM25/OpenSearch for sparse retrieval, a cross-encoder reranker, a local vLLM or llama.cpp model server, FastAPI for serving, Docker for packaging, and an evaluation set with retrieval and generation metrics. The demo backend intentionally stays lightweight so the learning notebooks work on a CPU without credentials.

See `docs/` for architecture decisions and `docs/deployment.md` for the production upgrade path.
