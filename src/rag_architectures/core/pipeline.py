from dataclasses import dataclass
import re
from .store import InMemoryStore
from .types import Document


class DemoLLM:
    """API-key-free deterministic generator; swap for a local open-weight model."""

    def generate(self, question, contexts):
        if not contexts:
            return "I could not find supporting context."
        words = {w for w in re.findall(r"\w+", question.lower()) if len(w) > 3}
        hits = []
        for context in contexts:
            hits += [
                sentence.strip()
                for sentence in re.split(r"(?<=[.!?])\s+", context.text)
                if any(word in sentence.lower() for word in words)
            ]
        return " ".join(hits[:3]) or contexts[0].text


@dataclass
class RAGResult:
    answer: str
    retrieved: list
    trace: dict


def ingest_documents(documents):
    """Stage 1: accept source documents into the RAG pipeline."""
    return [doc if isinstance(doc, Document) else Document(str(i), str(doc)) for i, doc in enumerate(documents)]


def parse_documents(documents):
    """Stage 2: normalize parsed document text."""
    parsed = []
    for doc in documents:
        text = re.sub(r"\s+", " ", doc.text).strip()
        parsed.append(Document(doc.id, text, dict(doc.metadata)))
    return parsed


def chunk_documents(documents, chunk_size=80, overlap=15):
    """Stage 3: split documents into overlapping retrieval chunks."""
    chunks = []
    step = max(1, chunk_size - overlap)
    for doc in documents:
        words = doc.text.split()
        if not words:
            continue
        for start in range(0, len(words), step):
            part = words[start:start + chunk_size]
            if not part:
                break
            metadata = dict(doc.metadata)
            metadata.update({"parent_id": doc.id, "chunk_index": len(chunks)})
            chunks.append(Document(f"{doc.id}:chunk:{start}", " ".join(part), metadata))
            if start + chunk_size >= len(words):
                break
    return chunks


def index_documents(chunks):
    """Stage 4: build the retrieval index."""
    return InMemoryStore(chunks)


def evaluate_answer(question, answer, retrieved):
    """Stage 8: lightweight deterministic evaluation for the teaching pipeline."""
    q_terms = {w for w in re.findall(r"\w+", question.lower()) if len(w) > 3}
    answer_terms = set(re.findall(r"\w+", answer.lower()))
    supported_terms = set()
    for hit in retrieved:
        supported_terms.update(re.findall(r"\w+", hit.document.text.lower()))
    return {
        "retrieval_count": len(retrieved),
        "answer_non_empty": bool(answer.strip()),
        "grounded_term_overlap": round(
            len(answer_terms & supported_terms) / max(1, len(answer_terms)), 3
        ),
        "query_coverage": round(
            len(q_terms & answer_terms) / max(1, len(q_terms)), 3
        ),
    }


def docs_from_pairs(pairs):
    return [Document(str(i), text, {"title": title}) for i, (title, text) in enumerate(pairs)]


class BaseRAG:
    name = "base"
    optional_stages = ("transform", "rerank", "verify")

    def __init__(self, documents=None, llm=None):
        self.ingested_documents = ingest_documents(list(documents or []))
        self.parsed_documents = parse_documents(self.ingested_documents)
        self.chunks = chunk_documents(self.parsed_documents)
        self.documents = self.chunks
        self.store = index_documents(self.chunks)
        self.llm = llm or DemoLLM()

    def retrieve(self, query, k=4):
        return self.store.search(query, k)

    def run(self, query, k=4):
        transformed_query = self.transform_query(query)
        hits = self.retrieve(transformed_query, k)
        hits = self.rerank(transformed_query, hits, k)
        hits = self.verify(transformed_query, hits, k)
        answer = self.llm.generate(query, [h.document for h in hits])
        evaluation = evaluate_answer(query, answer, hits)
        optional = {
            "transform": self.transform_detail(query, transformed_query),
            "rerank": self.rerank_detail(),
            "verify": self.verify_detail(),
        }
        pipeline = [
            {"step": "ingest", "status": "complete", "detail": f"{len(self.ingested_documents)} source documents"},
            {"step": "parse", "status": "complete", "detail": f"{len(self.parsed_documents)} parsed documents"},
            {"step": "chunk", "status": "complete", "detail": f"{len(self.chunks)} retrieval chunks"},
            {"step": "index", "status": "complete", "detail": type(self.store).__name__},
            {"step": "retrieve", "status": "complete", "detail": f"{len(hits)} final candidates"},
            {"step": "optional transform/rerank/verify", "status": "complete", "detail": optional},
            {"step": "generate", "status": "complete", "detail": type(self.llm).__name__},
            {"step": "evaluate", "status": "complete", "detail": evaluation},
        ]
        return RAGResult(
            answer=answer,
            retrieved=hits,
            trace={
                "architecture": self.name,
                "k": k,
                "pipeline": pipeline,
                "optional_stages": optional,
                "evaluation": evaluation,
            },
        )

    def transform_query(self, query):
        return query

    def transform_detail(self, original, transformed):
        return "skipped (query unchanged)" if original == transformed else f"{original!r} → {transformed!r}"

    def rerank(self, query, hits, k):
        return hits[:k]

    def rerank_detail(self):
        return "skipped (architecture does not rerank)"

    def verify(self, query, hits, k):
        return hits[:k]

    def verify_detail(self):
        return "skipped (architecture does not verify)"
