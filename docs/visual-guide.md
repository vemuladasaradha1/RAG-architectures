# RAG Architectures Visual Guide

The complete visual guide accompanies the 16 architecture notebooks.

Every notebook now follows the same explicit lifecycle:

**ingest → parse → chunk → index → retrieve → optional transform / rerank / verify → generate → evaluate**

The architecture-specific logic changes mainly in retrieval and the optional control stages, while the other stages remain explicit and testable.

Visual asset:

`../assets/rag-architectures-complete-visual-guide.jpg`

Data source used by the notebooks:

`../data/sample/mtech_quantum_project.html`
