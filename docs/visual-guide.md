# RAG Architectures Visual Guide

The complete visual guide covers all 16 architectures and is intended as a companion to the architecture notebooks.

## Canonical pipeline

Every architecture should be understood through the common lifecycle:

**ingest → parse → chunk → index → retrieve → optional transform / rerank / verify → generate → evaluate**

The architecture-specific differences happen mainly during retrieval and the optional transform/rerank/verify stage; ingestion, parsing, chunking, indexing, generation, and evaluation remain explicit parts of the end-to-end system.

## Visual asset

- `../assets/rag-architectures-complete-visual-guide.jpg`

Use the visual guide together with the corresponding notebook in `notebooks/` for the implementation walkthrough.
