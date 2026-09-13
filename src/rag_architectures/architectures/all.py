from __future__ import annotations

from collections import OrderedDict, defaultdict
import re

from ..core.pipeline import base_pipeline, lexical_score


def _dense(store, _chunks, query):
    return store.similarity_search(query, k=6)


def _sparse(_store, chunks, query):
    from langchain_community.retrievers import BM25Retriever

    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = 6
    return retriever.invoke(query)


def _hybrid(store, chunks, query):
    dense = _dense(store, chunks, query)
    sparse = _sparse(store, chunks, query)
    scores = defaultdict(float)
    by_id = {}
    for rank, document in enumerate(dense):
        key = str(document.metadata.get("chunk_id", id(document)))
        scores[key] += 1.0 / (rank + 1)
        by_id[key] = document
    for rank, document in enumerate(sparse):
        key = str(document.metadata.get("chunk_id", id(document)))
        scores[key] += 1.0 / (rank + 1)
        by_id[key] = document
    return [by_id[key] for key, _ in sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:6]]


def _rerank(query, documents):
    return sorted(documents, key=lambda doc: lexical_score(query, doc.page_content), reverse=True)


def _verify(query, documents):
    confidence = max((lexical_score(query, doc.page_content) for doc in documents), default=0.0)
    return {"confidence": round(confidence, 3), "passed": confidence >= 0.12}


def _multi_retrieve(store, chunks, variants):
    documents = {}
    scores = defaultdict(float)
    for variant in variants:
        for rank, document in enumerate(_dense(store, chunks, variant)):
            key = str(document.metadata.get("chunk_id", id(document)))
            documents[key] = document
            scores[key] += 1.0 / (rank + 1)
    return [documents[key] for key, _ in sorted(scores.items(), key=lambda pair: pair[1], reverse=True)[:6]]


def _entities(documents):
    words = re.findall(r"\b[A-Z][A-Za-z'-]{2,}\b", " ".join(d.page_content for d in documents))
    return sorted(set(words))


class BasicRAG:
    name = "basic"

    def run(self, raw_html, query):
        return base_pipeline(self.name, raw_html, query, _dense)


class DenseRAG(BasicRAG):
    name = "dense"


class SparseRAG(BasicRAG):
    name = "sparse"

    def run(self, raw_html, query):
        return base_pipeline(self.name, raw_html, query, _sparse)


class HybridRAG(BasicRAG):
    name = "hybrid"

    def run(self, raw_html, query):
        return base_pipeline(self.name, raw_html, query, _hybrid)


class RerankingRAG(BasicRAG):
    name = "reranking"

    def run(self, raw_html, query):
        return base_pipeline(self.name, raw_html, query, _hybrid, rerank_fn=_rerank)


class MultiQueryRAG(BasicRAG):
    name = "multi-query"

    def run(self, raw_html, query):
        variants = [
            query,
            f"Explain {query}",
            f"What are the key details about {query}",
            f"Describe {query} in the source document",
        ]
        return base_pipeline(
            self.name,
            raw_html,
            query,
            lambda store, chunks, _query: _multi_retrieve(store, chunks, variants),
            transform_fn=lambda original: " | ".join(variants),
        )


class HyDERAG(BasicRAG):
    name = "hyde"

    def run(self, raw_html, query):
        hypothetical = f"A useful answer passage would discuss: {query}"
        return base_pipeline(self.name, raw_html, query, _dense, transform_fn=lambda _: hypothetical)


class ContextualRAG(BasicRAG):
    name = "contextual"

    def run(self, raw_html, query):
        return base_pipeline(
            self.name,
            raw_html,
            query,
            _dense,
            transform_fn=lambda q: f"{q} surrounding section context metadata",
        )


class ParentChildRAG(BasicRAG):
    name = "parent-child"

    def run(self, raw_html, query):
        result = base_pipeline(self.name, raw_html, query, _dense)
        if result.retrieved:
            parent_ids = {doc.metadata.get("header_1") for doc in result.retrieved}
            result.trace.add("parent_context", selected_parent_sections=sorted(x for x in parent_ids if x))
        return result


class HierarchicalRAG(BasicRAG):
    name = "hierarchical"

    def run(self, raw_html, query):
        return base_pipeline(
            self.name,
            raw_html,
            query,
            _dense,
            transform_fn=lambda q: f"section summary topic {q}",
        )


class MultiHopRAG(BasicRAG):
    name = "multi-hop"

    def run(self, raw_html, query):
        result = base_pipeline(self.name, raw_html, query, _dense)
        if result.retrieved:
            evidence = " ".join(doc.page_content[:200] for doc in result.retrieved[:2])
            result.trace.add("hop_1", evidence=evidence)
            result.trace.add("hop_2", expanded_query=f"{query} {evidence}")
        return result


class GraphRAG(BasicRAG):
    name = "graph"

    def run(self, raw_html, query):
        result = base_pipeline(self.name, raw_html, query, _dense)
        result.trace.add("graph_build", entities=_entities(result.retrieved), relationships="entity co-occurrence")
        result.trace.add("graph_retrieve", strategy="entity-aware expansion hook")
        return result


class CorrectiveRAG(BasicRAG):
    name = "corrective"

    def run(self, raw_html, query):
        return base_pipeline(self.name, raw_html, query, _dense, verify_fn=_verify)


class SelfRAG(BasicRAG):
    name = "self-rag"

    def run(self, raw_html, query):
        return base_pipeline(self.name, raw_html, query, _dense, verify_fn=_verify)


class AdaptiveRAG(BasicRAG):
    name = "adaptive"

    def run(self, raw_html, query):
        retrieval = _sparse if len(query.split()) <= 7 else _dense
        return base_pipeline(self.name, raw_html, query, retrieval)


class AgenticRAG(BasicRAG):
    name = "agentic"

    def run(self, raw_html, query):
        result = base_pipeline(self.name, raw_html, query, _hybrid, rerank_fn=_rerank, verify_fn=_verify)
        result.trace.add("agent_plan", steps=["classify", "retrieve", "rerank", "verify", "generate", "evaluate"])
        return result


ARCHITECTURES = OrderedDict(
    [
        ("01_basic_rag", BasicRAG),
        ("02_dense_rag", DenseRAG),
        ("03_sparse_rag", SparseRAG),
        ("04_hybrid_rag", HybridRAG),
        ("05_reranking_rag", RerankingRAG),
        ("06_multi_query_rag", MultiQueryRAG),
        ("07_hyde", HyDERAG),
        ("08_contextual_rag", ContextualRAG),
        ("09_parent_child_rag", ParentChildRAG),
        ("10_hierarchical_rag", HierarchicalRAG),
        ("11_multihop_rag", MultiHopRAG),
        ("12_graphrag", GraphRAG),
        ("13_corrective_rag", CorrectiveRAG),
        ("14_self_rag", SelfRAG),
        ("15_adaptive_rag", AdaptiveRAG),
        ("16_agentic_rag", AgenticRAG),
    ]
)
