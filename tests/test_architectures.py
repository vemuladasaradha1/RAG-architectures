from rag_architectures.core.pipeline import docs_from_pairs
from rag_architectures.architectures.all import ARCHITECTURES

DOCS=docs_from_pairs([('RAG','Retrieval augmented generation retrieves evidence before generation.'),('Graph','Knowledge graphs connect entities and relationships.')])

def test_every_architecture_runs():
    for name, cls in ARCHITECTURES.items():
        result=cls(DOCS).run('What does retrieval augmented generation retrieve?')
        assert result.answer
        assert result.trace['architecture']
