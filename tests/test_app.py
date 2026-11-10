"""The Streamlit app, run headlessly with streamlit.testing.

AppTest cannot upload files, so tests either put a ready index into session
state or fake the uploader and the button click (see the upload tests).
"""
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import generation.generator
import generation.models
import pipeline
from conftest import DEFAULT_SETTINGS, FakeLLM, make_pdf

APP = str(Path(__file__).parent.parent / "app.py")
LIVE_CHAT_MODELS = {
    "openai": ["gpt-9", "gpt-4o-mini", "gpt-4o"],
    "claude": ["claude-next-1", "claude-haiku-5-5"],
}
LIVE_EMBEDDING_MODELS = ["text-embedding-9", "text-embedding-3-small"]


@pytest.fixture(autouse=True)
def no_real_apis(monkeypatch):
    """The app builds its LLM and its model lists through these; replace them all."""
    monkeypatch.setattr(generation.generator, "LLM", FakeLLM)
    monkeypatch.setattr(generation.models, "fetch_chat_models", lambda provider, api_key: LIVE_CHAT_MODELS[provider])
    monkeypatch.setattr(generation.models, "fetch_embedding_models", lambda api_key: LIVE_EMBEDDING_MODELS)
    st.cache_data.clear()


def run_app(**state) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=180)
    for key, value in state.items():
        at.session_state[key] = value
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def rerun(at: AppTest) -> AppTest:
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def ask(at: AppTest, question: str) -> dict:
    """Type a question and return the assistant's message."""
    at.chat_input[0].set_value(question)
    rerun(at)
    return at.session_state["messages"][-1]


def warnings_of(at: AppTest) -> str:
    return " | ".join(w.value for w in at.warning)


@pytest.fixture
def indexed(build):
    """An app with the test PDF already indexed (Hierarchical, dense) and no API key."""
    index, _ = build(chunk_strategy="Hierarchical")
    return lambda **state: run_app(index=index, doc_name="test.pdf", **state)


# first run ----------------------------------------------------------------
def test_starts_without_keys_or_document():
    at = run_app()
    assert at.chat_input[0].disabled
    assert at.selectbox(key="query_strategy").options == ["None (raw question)"]
    assert at.selectbox(key="model").options == ["gpt-4o-mini", "gpt-4o"]  # built-in list


@pytest.mark.parametrize("mode", ["Dense", "Sparse", "Hybrid"])
def test_every_retrieval_mode_renders(mode):
    at = run_app()
    at.selectbox(key="retrieval_mode").set_value(mode)
    rerun(at)
    shown = {widget.key for widget in at.selectbox}
    assert ("dense_model" in shown) == (mode != "Sparse")
    assert ("bm25_language" in shown) == (mode != "Dense")
    assert bool(at.slider) == (mode == "Hybrid")


def test_openai_embedding_without_key_warns_instead_of_indexing():
    at = run_app(dense_model="OpenAI Embedding")
    assert "OpenAI embeddings need an OpenAI API key" in warnings_of(at)


def test_invalid_chunk_sizes_show_an_error():
    at = run_app(chunk_overlap=600)
    assert any("overlap" in e.value for e in at.error)


# asking questions ---------------------------------------------------------
def test_question_without_llm_shows_chunks_only(indexed):
    at = indexed()
    message = ask(at, "What is an event horizon?")
    assert message["chunks"][0]["page"] == 3 and message["content"] is None
    assert any("showing retrieved chunks only" in i.value for i in at.info)


@pytest.mark.parametrize("strategy", ["None (raw question)", "Rewrite", "HyDE", "Decompose"])
def test_question_with_llm_and_each_query_strategy(indexed, strategy):
    at = indexed(openai_api_key="fake", query_strategy=strategy)
    message = ask(at, "plants and revolution?")
    assert message["content"] == "Fake answer [Source 1]"
    assert at.session_state["last_debug"]["query_strategy"] == strategy


def test_rerank_and_post_retrieval_steps_from_the_sidebar(indexed):
    at = indexed(openai_api_key="fake")
    at.toggle(key="rerank").set_value(True)
    for key in ("corrective_rag", "contextual_compression", "chunk_positioning"):
        at.checkbox(key=key).set_value(True)
    rerun(at)
    ask(at, "event horizon?")
    debug = at.session_state["last_debug"]
    skipped = [step["name"] for step in debug["steps"] if not step["ran"]]
    assert skipped == ["Query strategy"]  # left on "None"
    assert debug["dropped_chunks"] and debug["chunks"][0]["compressed_text"]


def test_llm_failure_becomes_a_chat_message(indexed, monkeypatch):
    class BrokenLLM(FakeLLM):
        def complete(self, *args, **kwargs):
            raise RuntimeError("invalid api key")

    monkeypatch.setattr(generation.generator, "LLM", BrokenLLM)
    at = indexed(openai_api_key="fake")
    message = ask(at, "event horizon?")
    assert message["error"] == "RuntimeError: invalid api key"
    assert any("invalid api key" in e.value for e in at.error)


def test_llm_features_switch_off_when_the_key_is_removed(indexed):
    at = indexed(openai_api_key="fake", corrective_rag=True, query_strategy="Rewrite")
    at.text_input(key="openai_api_key").set_value("")
    rerun(at)
    assert at.session_state["corrective_rag"] is False
    assert at.session_state["query_strategy"] == "None (raw question)"
    assert ask(at, "event horizon?")["content"] is None


# index state --------------------------------------------------------------
def test_changing_an_index_setting_asks_for_a_reindex(indexed):
    at = indexed(chunk_strategy="Hierarchical", chunk_size=200, chunk_parent_size=400, chunk_overlap=20)
    assert "settings changed" not in warnings_of(at)

    at.number_input(key="chunk_size").set_value(250)
    rerun(at)
    assert "settings changed since indexing" in warnings_of(at)
    # the old index keeps answering until then (this used to crash)
    assert ask(at, "event horizon?")["chunks"]


def test_query_time_setting_does_not_ask_for_a_reindex(indexed):
    at = indexed(chunk_strategy="Hierarchical", chunk_size=200, chunk_parent_size=400, chunk_overlap=20)
    at.number_input(key="retrieval_n_chunks").set_value(2)
    rerun(at)
    assert "settings changed" not in warnings_of(at)
    assert len(ask(at, "event horizon?")["chunks"]) == 2


def test_question_index_hides_hyde(build):
    index, _ = build(question_index=True, retrieval_mode="Hybrid")
    at = run_app(index=index, doc_name="test.pdf", openai_api_key="fake")
    assert "HyDE" not in at.selectbox(key="query_strategy").options
    assert ask(at, "What is Photosynthesis?")["chunks"][0]["matched_question"]


# model lists --------------------------------------------------------------
def test_model_list_comes_from_the_provider_when_a_key_is_set():
    at = run_app(openai_api_key="fake")
    assert at.selectbox(key="model").options == LIVE_CHAT_MODELS["openai"]
    assert at.session_state["model"] == "gpt-4o-mini"  # default stays the built-in first choice


def test_switching_provider_switches_the_model_list():
    at = run_app(openai_api_key="fake", anthropic_api_key="fake")
    at.selectbox(key="provider").set_value("claude")
    rerun(at)
    assert at.selectbox(key="model").options == LIVE_CHAT_MODELS["claude"]
    assert at.session_state["model"] == "claude-haiku-5-5"


def test_model_list_falls_back_when_the_provider_cannot_be_reached(monkeypatch):
    def unreachable(provider, api_key):
        raise ConnectionError("offline")

    monkeypatch.setattr(generation.models, "fetch_chat_models", unreachable)
    at = run_app(openai_api_key="fake")
    assert at.selectbox(key="model").options == ["gpt-4o-mini", "gpt-4o"]
    assert any("could not load the live one (ConnectionError)" in c.value for c in at.caption)


def test_embedding_model_list_is_live_too():
    at = run_app(openai_api_key="fake", dense_model="OpenAI Embedding")
    assert at.selectbox(key="openai_embedding_model_name").options == LIVE_EMBEDDING_MODELS
    assert at.session_state["openai_embedding_model_name"] == "text-embedding-3-small"


# upload and index ---------------------------------------------------------
class FakeUpload:
    name = "uploaded.pdf"

    def __init__(self, data: bytes):
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


@pytest.fixture
def upload_and_index(monkeypatch):
    """upload_and_index(pdf_bytes, **state): run the app as if the file was
    uploaded and 'Index document' clicked once."""

    def _run(data: bytes, **state) -> AppTest:
        clicks = {"left": 1}

        def fake_button(label, *args, **kwargs):
            if label == "Index document" and not kwargs.get("disabled") and clicks["left"]:
                clicks["left"] -= 1
                return True
            return False

        monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: FakeUpload(data))
        monkeypatch.setattr(st, "button", fake_button)
        return run_app(**state)

    return _run


def test_index_button_builds_an_index_and_chat_works(upload_and_index, pdf_bytes):
    at = upload_and_index(pdf_bytes)
    index = at.session_state["index"]
    assert index["chunk_count"] > 0 and at.session_state["doc_name"] == "uploaded.pdf"
    assert index["config"] == pipeline.index_config({**DEFAULT_SETTINGS, "chunk_strategy": "Fixed size", "chunk_size": 500, "chunk_overlap": 30})
    assert not at.chat_input[0].disabled
    assert ask(at, "event horizon?")["chunks"][0]["page"] == 3


def test_index_button_with_hybrid_question_index(upload_and_index, pdf_bytes):
    at = upload_and_index(pdf_bytes, openai_api_key="fake", retrieval_mode="Hybrid", question_index=True)
    index = at.session_state["index"]
    assert index["dense"] and index["sparse"] and index["config"]["question_index"]
    assert len(index["llm_usage"]) == index["chunk_count"]


@pytest.mark.parametrize("data, state, message", [
    (make_pdf([""]), {}, "No text could be extracted"),
    (b"this is not a pdf", {}, "Indexing failed"),
    (None, {"chunk_overlap": 600}, "overlap must be smaller"),
])
def test_indexing_problems_are_reported_not_raised(upload_and_index, pdf_bytes, data, state, message):
    at = upload_and_index(pdf_bytes if data is None else data, **state)
    assert at.session_state["index"] is None
    assert any(message in e.value for e in at.error)
