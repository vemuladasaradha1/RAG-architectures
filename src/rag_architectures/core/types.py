from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True)
class Document:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class RetrievedDocument:
    document: Document
    score: float
    source: str = 'dense'
