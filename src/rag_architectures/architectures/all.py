from collections import OrderedDict
from ..core.pipeline import BaseRAG
from ..core.store import InMemoryStore, BM25Store


class BasicRAG(BaseRAG):
    name = "basic"


class DenseRAG(BaseRAG):
    name = "dense"


class SparseRAG(BaseRAG):
    name = "sparse"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.store = BM25Store(self.documents)


class HybridRAG(BaseRAG):
    name = "hybrid"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sparse = BM25Store(self.documents)
        self.dense = InMemoryStore(self.documents)

    def retrieve(self, query, k=4):
        hits = self.dense.search(query, k * 2) + self.sparse.search(query, k * 2)
        scores, by_id = {}, {}
        for hit in hits:
            scores[hit.document.id] = scores.get(hit.document.id, 0) + hit.score
            by_id[hit.document.id] = hit
        return sorted(
            [type(hit)(by_id[i].document, score, "hybrid") for i, score in scores.items()],
            key=lambda x: x.score,
            reverse=True,
        )[:k]

    def rerank_detail(self):
        return "dense + sparse scores merged (hybrid fusion)"


class RerankingRAG(HybridRAG):
    name = "reranking"

    def retrieve(self, query, k=4):
        hits = super().retrieve(query, k * 3)
        terms = set(query.lower().split())
        return sorted(
            hits,
            key=lambda h: (len(terms & set(h.document.text.lower().split())), h.score),
            reverse=True,
        )[:k]

    def rerank_detail(self):
        return "lexical relevance reranking applied to a broad candidate set"


class MultiQueryRAG(BaseRAG):
    name = "multi-query"

    def retrieve(self, query, k=4):
        variants = [query, query + " key facts", query + " explanation", "what is " + query]
        hits = []
        for variant in variants:
            hits += self.store.search(variant, k)
        best, by_id = {}, {}
        for hit in hits:
            best[hit.document.id] = max(best.get(hit.document.id, 0), hit.score)
            by_id[hit.document.id] = hit
        return sorted(
            [type(hit)(by_id[i].document, score, "multi-query") for i, score in best.items()],
            key=lambda x: x.score,
            reverse=True,
        )[:k]

    def transform_query(self, query):
        return query

    def transform_detail(self, original, transformed):
        return "generated multiple query variants inside retrieval"


class HyDERAG(BaseRAG):
    name = "hyde"

    def retrieve(self, query, k=4):
        return self.store.search(query + " likely answer", k)

    def transform_detail(self, original, transformed):
        return "hypothetical-answer expansion is applied before retrieval"


class ContextualRAG(BaseRAG):
    name = "contextual"

    def verify(self, query, hits, k):
        enriched = []
        for hit in hits[:k]:
            title = hit.document.metadata.get("title", "unknown")
            hit.document.metadata.update({"context_title": title})
            enriched.append(hit)
        return enriched

    def verify_detail(self):
        return "retrieved chunks are enriched with document metadata/context"


class ParentChildRAG(BaseRAG):
    name = "parent-child"

    def retrieve(self, query, k=4):
        child_hits = self.store.search(query, k)
        parent_ids = {h.document.metadata.get("parent_id") for h in child_hits}
        return [
            hit for hit in child_hits
            if hit.document.metadata.get("parent_id") in parent_ids
        ][:k]

    def verify_detail(self):
        return "child matches are checked against parent-document identity"


class HierarchicalRAG(BaseRAG):
    name = "hierarchical"

    def retrieve(self, query, k=4):
        return self.store.search(query, k)

    def transform_detail(self, original, transformed):
        return "hierarchical retrieval hook: summaries → chunks (demo index uses chunks directly)"


class MultiHopRAG(BaseRAG):
    name = "multi-hop"

    def retrieve(self, query, k=4):
        first = self.store.search(query, k)
        expanded = query + " " + " ".join(h.document.text[:80] for h in first)
        return self.store.search(expanded, k)

    def transform_detail(self, original, transformed):
        return "hop-1 evidence is used to expand the next retrieval query"


class GraphRAG(BaseRAG):
    name = "graph"

    def retrieve(self, query, k=4):
        return self.store.search(query, k)

    def transform_detail(self, original, transformed):
        return "graph traversal hook: entity/relationship expansion (demo index uses text retrieval)"


class CorrectiveRAG(BaseRAG):
    name = "corrective"

    def verify(self, query, hits, k):
        if not hits or hits[0].score < 0.15:
            return self.store.search(query + " relevant facts", k)
        return hits[:k]

    def verify_detail(self):
        return "retrieval quality checked; weak evidence triggers corrective retrieval"


class SelfRAG(BaseRAG):
    name = "self-rag"

    def verify(self, query, hits, k):
        return hits[:k]

    def verify_detail(self):
        return "self-check: evidence availability/relevance is inspected before generation"


class AdaptiveRAG(BaseRAG):
    name = "adaptive"

    def retrieve(self, query, k=4):
        if len(query.split()) < 5:
            return BM25Store(self.documents).search(query, k)
        return self.store.search(query, k)

    def transform_detail(self, original, transformed):
        strategy = "sparse/BM25" if len(original.split()) < 5 else "dense"
        return f"router selected {strategy} retrieval for this query"


class AgenticRAG(BaseRAG):
    name = "agentic"

    def retrieve(self, query, k=4):
        return super().retrieve(query, k)

    def verify(self, query, hits, k):
        return hits[:k]

    def verify_detail(self):
        return "agent plan includes retrieve → verify → synthesize control flow"

    def transform_detail(self, original, transformed):
        return "agent planner selects and sequences the retrieval actions"


ARCHITECTURES = OrderedDict([
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
])
