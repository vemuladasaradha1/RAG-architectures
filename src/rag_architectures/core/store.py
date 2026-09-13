from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from .pipeline import HashEmbeddings


class LangChainVectorStore:
    """Offline LangChain vector store used by the shared pipeline."""

    def __init__(self, documents: list[Document] | None = None) -> None:
        self.store = InMemoryVectorStore(HashEmbeddings())
        if documents:
            self.store.add_documents(documents)

    def add_documents(self, documents: list[Document]) -> None:
        self.store.add_documents(documents)

    def search(self, query: str, k: int = 5) -> list[Document]:
        return self.store.similarity_search(query, k=k)
