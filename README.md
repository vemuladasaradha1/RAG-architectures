# RAG Architectures

A LangChain-based learning and engineering repository that implements 16 major RAG architecture patterns around one explicit lifecycle:

**ingest → parse → chunk → index → retrieve → optional transform / rerank / verify → generate → evaluate**

The notebooks use `data/sample/mtech_quantum_project.html` as the canonical offline corpus. Each notebook exposes the common pipeline and then highlights the architecture-specific behavior.

## 16 architectures

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
13. Corrective RAG (CRAG)
14. Self-RAG
15. Adaptive RAG
16. Agentic RAG

## LangChain stack

The repository is built on LangChain 1.x. It uses LangChain Core documents, runnable pipelines, vector-store abstractions, LangChain text splitters, and the Community BM25 retriever. LangChain's HTML header splitter and recursive splitter are used for parse/chunk stages; `InMemoryVectorStore` handles the offline index; and a `RunnableLambda` provides a deterministic offline generator. This keeps every notebook executable without API keys while leaving clean seams for local open-weight models and production vector databases.

## Data source

`data/sample/mtech_quantum_project.html` is a repository copy derived from the user-supplied M.Tech Quantum ML project HTML and is used by all 16 notebooks as the retrieval corpus.

## Run

```bash
python -m pip install -e ".[dev,notebooks]"
pytest -q
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```

For real semantic embeddings and model inference, use the production extras and replace the offline `HashEmbeddings` and deterministic generator with local open-weight models.
