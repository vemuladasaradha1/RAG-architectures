# Architecture overview

All 16 examples share the same explicit LangChain-oriented end-to-end lifecycle:

**ingest → parse → chunk → index → retrieve → optional transform / rerank / verify → generate → evaluate**

The variation is deliberately isolated:

- Basic and Dense: baseline vector retrieval.
- Sparse: BM25 lexical retrieval.
- Hybrid: dense + sparse fusion.
- Reranking: broad candidate retrieval followed by relevance ordering.
- Multi-query and HyDE: query transformation before retrieval.
- Contextual, Parent-child and Hierarchical: context selection and expansion.
- Multi-hop and GraphRAG: iterative or relationship-aware retrieval.
- Corrective and Self-RAG: verification/reflection around evidence.
- Adaptive and Agentic: routing and orchestration across retrieval actions.

Every notebook makes all lifecycle stages visible so learners can compare what changes, what it costs, and where an architecture can fail.