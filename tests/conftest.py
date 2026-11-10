"""Shared fixtures. Nothing in the test suite calls a real LLM or embedding API:
the LLM is replaced by FakeLLM and only local models are used.

The first run downloads all-MiniLM-L6-v2 and the reranker from Hugging Face.
"""
import json
import os
import tempfile

# Before any project import: ignore the keys in .env (load_dotenv does not
# override variables that are already set) and keep Chroma out of .chroma_db.
os.environ["ANTHROPIC_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""
os.environ["CHROMA_PATH"] = tempfile.mkdtemp(prefix="rag-explorer-tests-")

import fitz
import pytest

import pipeline
from ingestion.loader import load_pdf

# One topic per page, so the page number of a retrieved chunk says whether retrieval worked
TOPICS = {
    1: "Photosynthesis is the process by which green plants convert sunlight, water and carbon dioxide into glucose and oxygen. It happens in the chloroplasts.",
    2: "The French Revolution began in 1789 with the storming of the Bastille. It ended the monarchy of Louis XVI and led to the rise of Napoleon Bonaparte.",
    3: "A black hole is a region of spacetime where gravity is so strong that nothing, not even light, can escape. The boundary is called the event horizon.",
    4: "Sourdough bread is leavened with a starter of wild yeast and lactic acid bacteria. The dough ferments slowly, which gives the bread its sour taste.",
}

DEFAULT_SETTINGS = {
    "chunk_strategy": "Semantic",
    "chunk_size": 200,
    "chunk_parent_size": 400,
    "chunk_overlap": 20,
    "retrieval_mode": "Dense",
    "dense_model": "all-MiniLM-L6-v2",
    "openai_embedding_model_name": "text-embedding-3-small",
    "bm25_language": "english",
    "bm25_stemming": True,
    "bm25_stopwords_mode": "Language default",
    "bm25_stopwords_custom": "",
    "question_index": False,
    "questions_per_chunk": 2,
    "hybrid_dense_weight": 0.5,
    "retrieval_n_chunks": 4,
    "rerank": False,
    "rerank_candidates": 10,
    "query_strategy": "None (raw question)",
    "corrective_rag": False,
    "contextual_compression": False,
    "chunk_positioning": False,
    "max_tokens": 500,
}


class FakeLLM:
    """Stands in for generation.generator.LLM and answers by the label of the call."""

    def __init__(self, provider="openai", model="fake-model", api_key="fake"):
        self.provider, self.model, self.usage = provider, model, []

    def complete(self, prompt, max_tokens=2000, label="LLM call"):
        self.usage.append({"label": label, "input_tokens": len(prompt) // 4, "output_tokens": 10})
        if label == "Query decomposition":
            return 'Sure: ["What is photosynthesis?", "When did the French Revolution begin?"]'
        if label == "Query rewrite":
            return "How do black holes trap light?"
        if label == "HyDE document":
            return "Sourdough is made with a wild yeast starter and ferments slowly."
        if label == "Hypothetical questions":
            subject = prompt.split("Text:")[-1].strip().split()[0]
            return json.dumps([f"What is {subject}?", f"Tell me about {subject}"])
        if label == "Corrective RAG":
            return "[1]"
        if label == "Contextual compression":
            return "COMPRESSED"
        return "Fake answer [Source 1]"


def make_pdf(texts) -> bytes:
    doc = fitz.open()
    for text in texts:
        page = doc.new_page()
        if text:
            page.insert_textbox(fitz.Rect(50, 50, 550, 800), (text + "\n\n") * 3, fontsize=11)
    return doc.tobytes()


@pytest.fixture(scope="session")
def pdf_bytes() -> bytes:
    return make_pdf(TOPICS.values())


@pytest.fixture(scope="session")
def pages(pdf_bytes):
    return load_pdf(pdf_bytes)


@pytest.fixture(scope="session")
def build(pages):
    """build(**setting_overrides) -> (index, settings). Indexes are cached per
    config, since most tests only vary query-time settings."""
    cache = {}

    def _build(**overrides):
        settings = {**DEFAULT_SETTINGS, **overrides}
        config = pipeline.index_config(settings)
        key = json.dumps(config, sort_keys=True)
        if key not in cache:
            cache[key] = pipeline.build_index(
                pages, config, session_id=f"test-{len(cache)}",
                llm=FakeLLM() if settings["question_index"] else None,
                log=lambda message: None,
            )
        return cache[key], settings

    return _build


@pytest.fixture(scope="session")
def ask(build):
    """ask(question, llm=None, **setting_overrides) -> result of answer_question."""

    def _ask(question, llm=None, **overrides):
        index, settings = build(**overrides)
        return pipeline.answer_question(question, index, settings, llm)

    return _ask


def pages_of(result) -> list[int]:
    return [chunk["page"] for chunk in result["chunks"]]


def steps_of(result) -> dict:
    """{step name: ran} of the process log."""
    return {step["name"]: step["ran"] for step in result["debug"]["steps"]}
