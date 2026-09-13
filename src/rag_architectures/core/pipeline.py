from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
import re
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableLambda
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import HTMLHeaderTextSplitter, RecursiveCharacterTextSplitter


class HashEmbeddings(Embeddings):
    """Deterministic offline embeddings for reproducible notebooks and CI."""

    def __init__(self, dimensions: int = 256) -> None:
        self.dimensions = dimensions

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for offset in range(0, min(8, len(digest) - 1), 2):
                index = int.from_bytes(digest[offset : offset + 2], "little") % self.dimensions
                vector[index] += 1.0 if digest[offset] % 2 else -1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@dataclass
class PipelineTrace:
    architecture: str
    stages: list[dict[str, Any]] = field(default_factory=list)

    def add(self, stage: str, **details: Any) -> None:
        self.stages.append({"stage": stage, **details})


@dataclass
class RAGResult:
    answer: str
    retrieved: list[Document]
    trace: PipelineTrace
    evaluation: dict[str, float]


def ingest_html(path: str) -> str:
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def parse_html(raw_html: str) -> list[Document]:
    splitter = HTMLHeaderTextSplitter(
        headers_to_split_on=[
            ("h1", "header_1"),
            ("h2", "header_2"),
            ("h3", "header_3"),
            ("h4", "header_4"),
        ]
    )
    documents = splitter.split_text(raw_html)
    return documents or [Document(page_content=raw_html, metadata={})]


def chunk_documents(
    documents: list[Document], chunk_size: int = 700, chunk_overlap: int = 100
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    for index, document in enumerate(chunks):
        document.metadata = {**document.metadata, "chunk_id": index}
    return chunks


def build_index(documents: list[Document]) -> InMemoryVectorStore:
    store = InMemoryVectorStore(HashEmbeddings())
    store.add_documents(documents)
    return store


def lexical_score(query: str, text: str) -> float:
    query_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
    text_tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
    return len(query_tokens & text_tokens) / max(1, len(query_tokens))


def extractive_generate(question: str, contexts: list[Document]) -> str:
    if not contexts:
        return "I could not find supporting evidence in the supplied document."
    terms = [token for token in re.findall(r"[a-z0-9]+", question.lower()) if len(token) > 3]
    sentences: list[str] = []
    for document in contexts:
        for sentence in re.split(r"(?<=[.!?])\s+", document.page_content.strip()):
            if any(term in sentence.lower() for term in terms):
                sentences.append(sentence)
    return " ".join(sentences[:4]).strip() or contexts[0].page_content[:800]


def evaluate_answer(answer: str, retrieved: list[Document]) -> dict[str, float]:
    grounded = 1.0 if retrieved and answer else 0.0
    answer_tokens = set(re.findall(r"[a-z0-9]+", answer.lower()))
    coverage = min(1.0, len(answer_tokens) / 40.0) if answer_tokens else 0.0
    return {"grounded": grounded, "answer_coverage": round(coverage, 3)}


def base_pipeline(
    architecture: str,
    raw_html: str,
    query: str,
    retrieval_fn,
    transform_fn=None,
    verify_fn=None,
    rerank_fn=None,
) -> RAGResult:
    trace = PipelineTrace(architecture)
    trace.add("ingest", source="mtech_quantum_project.html")
    parsed = parse_html(raw_html)
    trace.add("parse", documents=len(parsed))
    chunks = chunk_documents(parsed)
    trace.add("chunk", chunks=len(chunks), chunk_size=700, overlap=100)
    store = build_index(chunks)
    trace.add("index", backend="LangChain InMemoryVectorStore")

    working_query = transform_fn(query) if transform_fn else query
    if transform_fn:
        trace.add("transform", query=working_query)

    retrieved = retrieval_fn(store, chunks, working_query)
    trace.add("retrieve", count=len(retrieved))
    if rerank_fn:
        retrieved = rerank_fn(query, retrieved)
        trace.add("rerank", count=len(retrieved))
    if verify_fn:
        verification = verify_fn(query, retrieved)
        trace.add("verify", **verification)

    answer = extractive_generate(query, retrieved)
    trace.add("generate", generator="LangChain RunnableLambda + deterministic extractor")
    evaluation = evaluate_answer(answer, retrieved)
    trace.add("evaluate", **evaluation)
    return RAGResult(answer, retrieved, trace, evaluation)


def langchain_generator() -> RunnableLambda:
    return RunnableLambda(
        lambda payload: extractive_generate(payload["question"], payload["contexts"])
    )
