import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_evaluation import build_reference_block, classify_missing_info, get_llm_config


def test_build_reference_block_formats_sources():
    refs = build_reference_block([
        {"source": "sample.pdf", "page": 4, "text": "The company is remote-friendly."},
        {"source": "sample.pdf", "page": 6, "text": "Employees have ownership."},
    ])

    assert "sample.pdf" in refs
    assert "Page 4" in refs
    assert "Page 6" in refs


def test_classify_missing_info_marks_out_of_scope_questions():
    missing = classify_missing_info(
        "What is the capital of France?",
        "The document does not mention the capital of France."
    )

    assert missing["is_answerable"] is False
    assert "not available in the document" in missing["reason"].lower()


def test_get_llm_config_uses_groq_when_gemini_is_missing(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    monkeypatch.delenv("LLM_MODEL", raising=False)

    config = get_llm_config()

    assert config["provider"] == "groq"
    assert config["model"] == "openai/gpt-oss-20b"
