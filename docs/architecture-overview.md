# Architecture overview

All 16 architectures share: ingest → parse → chunk → index → retrieve → optional transform/rerank/verify → generate → evaluate.

The learning goal is to change one stage at a time and measure it. Hybrid combines sparse + semantic retrieval. Reranking improves ordering. Multi-query and HyDE transform queries. Parent-child and hierarchical designs improve context selection. Multi-hop, graph, corrective, self-reflective, adaptive and agentic designs add iterative reasoning or control flow.

For each notebook ask: What problem does it solve? What does it change? What does it cost? How does it fail? How should it be evaluated?
