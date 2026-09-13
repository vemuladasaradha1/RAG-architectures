from dataclasses import dataclass
from .store import InMemoryStore
from .types import Document

class DemoLLM:
    """API-key-free deterministic generator; swap for local Transformers/vLLM/llama.cpp."""
    def generate(self, question, contexts):
        if not contexts: return 'I could not find supporting context.'
        words={w for w in question.lower().split() if len(w)>3}; hits=[]
        for c in contexts:
            hits += [s.strip() for s in c.text.split('. ') if any(w in s.lower() for w in words)]
        return ' '.join(hits[:3]) or contexts[0].text

@dataclass
class RAGResult:
    answer: str
    retrieved: list
    trace: dict

class BaseRAG:
    name='base'
    def __init__(self, documents=None, llm=None):
        self.documents=list(documents or []); self.store=InMemoryStore(self.documents); self.llm=llm or DemoLLM()
    def retrieve(self, query, k=4): return self.store.search(query,k)
    def run(self, query, k=4):
        hits=self.retrieve(query,k); return RAGResult(self.llm.generate(query,[h.document for h in hits]),hits,{'architecture':self.name,'k':k})

def docs_from_pairs(pairs): return [Document(str(i),text,{'title':title}) for i,(title,text) in enumerate(pairs)]
