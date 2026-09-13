import re, math
from collections import Counter
from .types import RetrievedDocument

class InMemoryStore:
    """Dependency-light teaching backend; replace with Qdrant/FAISS in production."""
    def __init__(self, documents=None): self.documents=list(documents or [])
    @staticmethod
    def _tokens(text): return re.findall(r'[a-z0-9]+', text.lower())
    def add(self, documents): self.documents.extend(documents)
    def search(self, query, k=5):
        q=Counter(self._tokens(query)); out=[]; nq=sum(v*v for v in q.values())**.5
        for d in self.documents:
            c=Counter(self._tokens(d.text)); dot=sum(q[t]*c[t] for t in q); nd=sum(v*v for v in c.values())**.5
            out.append(RetrievedDocument(d,dot/(nq*nd) if nq and nd else 0.0,'dense-demo'))
        return sorted(out,key=lambda x:x.score,reverse=True)[:k]

class BM25Store(InMemoryStore):
    def search(self, query, k=5):
        q=set(self._tokens(query)); N=len(self.documents); avg=sum(len(self._tokens(d.text)) for d in self.documents)/(N or 1); out=[]; K1=1.5; B=.75
        for d in self.documents:
            toks=self._tokens(d.text); dl=len(toks); tf=Counter(toks); score=0.0
            for t in q:
                df=sum(t in self._tokens(x.text) for x in self.documents)
                if df:
                    idf=max(0,math.log((N-df+.5)/(df+.5)+1)); score+=idf*(tf[t]*(K1+1))/(tf[t]+K1*(1-B+B*dl/(avg or 1)))
            out.append(RetrievedDocument(d,score,'bm25'))
        return sorted(out,key=lambda x:x.score,reverse=True)[:k]
