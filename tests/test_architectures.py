from pathlib import Path

from rag_architectures.architectures.all import ARCHITECTURES


DATA_PATH = Path("data/sample/mtech_quantum_project.html")


def test_every_architecture_runs():
    raw_html = DATA_PATH.read_text(encoding="utf-8")
    assert len(ARCHITECTURES) == 16

    for cls in ARCHITECTURES.values():
        result = cls().run(
            raw_html,
            "What are the key implementation steps and expected outcomes in the M.Tech project?",
        )
        stages = [item["stage"] for item in result.trace.stages]
        assert stages[:4] == ["ingest", "parse", "chunk", "index"]
        assert "retrieve" in stages
        assert "generate" in stages
        assert "evaluate" in stages
        assert result.answer
