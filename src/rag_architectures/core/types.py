from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document


@dataclass(frozen=True)
class RetrievedDocument:
    document: Document
    score: float
    source: str = "dense"


Metadata = dict[str, Any]
