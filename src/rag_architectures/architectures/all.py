from collections import OrderedDict
from ..core.pipeline import BaseRAG
from ..core.store import InMemoryStore, BM25Store

class BasicRAG(BaseRAG): name='basic'
class DenseRAG(BaseRAG): name='dense'
class SparseRAG(BaseRAG):
    name='sparse'
    def __init__(self,*a,**kw): super().__init__(*a,**kw); self.store=BM25Store(self.documents)
class HybridRAG(BaseRAG):
    name='hybrid'
    def __init__(self,*a,**kw):
        super().__init__(*a,**kw); self.sparse=BM25Store(self.documents); self.dense=InMemoryStore(self.documents)
    def retrieve(self,q,k=4):
        hits=self.dense.search(q,k*2)+self.sparse.search(q,k*2); scores={}; by={}
        for h in hits: scores[h.document.id]=scores.get(h.document.id,0)+h.score; by[h.document.id]=h
        return sorted([type(h)(by[i].document,s,'hybrid') for i,s in scores.items()],key=lambda x:x.score,reverse=True)[:k]
class RerankingRAG(HybridRAG):
    name='reranking'
    def retrieve(self,q,k=4):
        hits=super().retrieve(q,k*3); qs=set(q.lower().split()); return sorted(hits,key=lambda h:(len(qs&set(h.document.text.lower().split())),h.score),reverse=True)[:k]
class MultiQueryRAG(BaseRAG):
    name='multi-query'
    def retrieve(self,q,k=4):
        hs=[]
        for v in [q,q+' key facts',q+' explanation','what is '+q]: hs+=self.store.search(v,k)
        best={}; by={}
        for h in hs: best[h.document.id]=max(best.get(h.document.id,0),h.score); by[h.document.id]=h
        return sorted([type(h)(by[i].document,s,'multi-query') for i,s in best.items()],key=lambda x:x.score,reverse=True)[:k]
class HyDERAG(BaseRAG):
    name='hyde'
    def retrieve(self,q,k=4): return self.store.search(q+' likely answer',k)
class ContextualRAG(BaseRAG): name='contextual'
class ParentChildRAG(BaseRAG): name='parent-child'
class HierarchicalRAG(BaseRAG): name='hierarchical'
class MultiHopRAG(BaseRAG):
    name='multi-hop'
    def retrieve(self,q,k=4):
        first=self.store.search(q,k); return self.store.search(q+' '+' '.join(h.document.text[:80] for h in first),k)
class GraphRAG(BaseRAG): name='graph'
class CorrectiveRAG(BaseRAG):
    name='corrective'
    def run(self,q,k=4):
        r=super().run(q,k); r.trace['corrected']=not r.retrieved or r.retrieved[0].score<.15; return r
class SelfRAG(BaseRAG):
    name='self-rag'
    def run(self,q,k=4):
        r=super().run(q,k); r.trace['self_check']=bool(r.retrieved) and r.retrieved[0].score>.1; return r
class AdaptiveRAG(BaseRAG):
    name='adaptive'
    def retrieve(self,q,k=4): return BM25Store(self.documents).search(q,k) if len(q.split())<5 else self.store.search(q,k)
class AgenticRAG(BaseRAG):
    name='agentic'
    def run(self,q,k=4):
        r=super().run(q,k); r.trace['agent_steps']=['classify','retrieve','verify','answer']; return r

ARCHITECTURES=OrderedDict([
('01_basic_rag',BasicRAG),('02_dense_rag',DenseRAG),('03_sparse_rag',SparseRAG),('04_hybrid_rag',HybridRAG),
('05_reranking_rag',RerankingRAG),('06_multi_query_rag',MultiQueryRAG),('07_hyde',HyDERAG),('08_contextual_rag',ContextualRAG),
('09_parent_child_rag',ParentChildRAG),('10_hierarchical_rag',HierarchicalRAG),('11_multihop_rag',MultiHopRAG),('12_graphrag',GraphRAG),
('13_corrective_rag',CorrectiveRAG),('14_self_rag',SelfRAG),('15_adaptive_rag',AdaptiveRAG),('16_agentic_rag',AgenticRAG)])
