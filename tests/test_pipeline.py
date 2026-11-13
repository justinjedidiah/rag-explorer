"""build_index() and answer_question() end to end, with local models and a fake LLM.

The test PDF has one topic per page (see conftest.TOPICS), so asserting on page
numbers checks that the right chunks were retrieved.
"""
import pytest

import pipeline
from conftest import DEFAULT_SETTINGS, FakeLLM, pages_of, steps_of

MODES = ["Dense", "Sparse", "Hybrid"]


# retrieval ----------------------------------------------------------------
@pytest.mark.parametrize("mode, score_type", [("Dense", "cosine"), ("Sparse", "bm25"), ("Hybrid", "hybrid")])
def test_each_mode_finds_the_right_page(ask, mode, score_type):
    result = ask("storming of the Bastille", retrieval_mode=mode)
    assert result["chunks"][0]["page"] == 2
    assert {c["score_type"] for c in result["chunks"]} == {score_type}
    assert result["answer"] is None  # no LLM given


def test_dense_understands_meaning_without_shared_keywords(ask):
    assert ask("How do plants make oxygen?")["chunks"][0]["page"] == 1


def test_sparse_returns_nothing_when_no_term_matches(ask):
    assert ask("zzzz qqqq", retrieval_mode="Sparse")["chunks"] == []


def test_hybrid_still_answers_when_bm25_matches_nothing(ask):
    result = ask("zzzz qqqq", retrieval_mode="Hybrid")
    assert result["chunks"] and all(c["sparse_score"] is None for c in result["chunks"])


def test_hybrid_weight_decides_which_side_counts(ask):
    question = "Bastille monarchy"
    dense_only = ask(question, retrieval_mode="Hybrid", hybrid_dense_weight=1.0)["chunks"]
    sparse_only = ask(question, retrieval_mode="Hybrid", hybrid_dense_weight=0.0)["chunks"]
    assert dense_only[0]["dense_score"] == max(c["dense_score"] for c in dense_only if c["dense_score"] is not None)
    assert sparse_only[0]["sparse_score"] == max(c["sparse_score"] for c in sparse_only if c["sparse_score"] is not None)
    assert all(0 <= c["score"] <= 1 for c in dense_only + sparse_only)


def test_hierarchical_returns_child_with_parent(ask):
    chunk = ask("event horizon", chunk_strategy="Hierarchical")["chunks"][0]
    assert chunk["page"] == 3 and chunk["text"] in chunk["parent_text"]


def test_no_duplicate_chunks(ask):
    for mode in MODES:
        ids = [c["chunk_id"] for c in ask("yeast and bacteria", retrieval_mode=mode)["chunks"]]
        assert len(ids) == len(set(ids))


def test_reranker_rescores_and_keeps_n(ask):
    result = ask("What gives sourdough its sour taste?", rerank=True)
    chunks = result["chunks"]
    assert len(chunks) == DEFAULT_SETTINGS["retrieval_n_chunks"]
    assert chunks[0]["page"] == 4 and chunks[0]["score_type"] == "rerank"
    assert chunks[0]["retrieval_score_type"] == "cosine" and "retrieval_score" in chunks[0]
    assert [c["score"] for c in chunks] == sorted((c["score"] for c in chunks), reverse=True)
    assert all(0 <= c["score"] <= 1 for c in chunks)
    assert steps_of(result)["Reranker"]


def test_reranker_with_no_candidates(ask):
    assert ask("zzzz qqqq", retrieval_mode="Sparse", rerank=True)["chunks"] == []


# query strategies ---------------------------------------------------------
def test_without_llm_the_strategy_is_ignored(ask):
    result = ask("How do plants make oxygen?", query_strategy="Rewrite")
    assert result["debug"]["query_strategy"] == "None (raw question)"
    assert steps_of(result) == {
        "Query strategy": False, "Dense retrieval": True, "Reranker": False, "Corrective RAG": False,
        "Contextual compression": False, "Chunk positioning": False, "Generation": False,
    }


def test_decompose_retrieves_for_every_sub_question(ask):
    # the fake LLM splits into a photosynthesis and a French Revolution question
    llm = FakeLLM()
    result = ask("plants and revolution?", llm=llm, query_strategy="Decompose")
    assert {1, 2} <= set(pages_of(result))
    assert len(result["debug"]["query_chunk_map"]) == 2
    assert [u["label"] for u in llm.usage] == ["Query decomposition", "Generation"]
    assert result["answer"] == "Fake answer [Source 1]"


@pytest.mark.parametrize("strategy, display_key, expected_page", [
    ("Rewrite", "rewritten", 3),            # fake rewrite asks about black holes
    ("HyDE", "hypothetical_document", 4),   # fake document is about sourdough
])
def test_rewrite_and_hyde_search_with_the_llm_text(ask, strategy, display_key, expected_page):
    result = ask("something vague", llm=FakeLLM(), query_strategy=strategy)
    assert result["chunks"][0]["page"] == expected_page
    assert result["debug"]["query_display"][display_key]


# hypothetical-questions index ---------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_question_index_maps_hits_back_to_chunks(build, ask, mode):
    index, _ = build(question_index=True, retrieval_mode=mode)
    assert len(index["llm_usage"]) == index["chunk_count"]  # one LLM call per chunk

    chunks = ask("What is Photosynthesis?", question_index=True, retrieval_mode=mode)["chunks"]
    top = chunks[0]
    assert top["page"] == 1
    assert "Photosynthesis" in top["matched_question"]
    assert top["text"].startswith("Photosynthesis is the process")  # the chunk, not the question
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)      # two questions per chunk, one result


def test_hyde_is_skipped_on_a_question_index(ask):
    llm = FakeLLM()
    result = ask("What is Photosynthesis?", llm=llm, question_index=True, query_strategy="HyDE")
    assert result["debug"]["query_strategy"] == "None (raw question)"
    assert "HyDE document" not in [u["label"] for u in llm.usage]
    assert result["chunks"][0]["page"] == 1


def test_question_index_needs_an_llm(pages):
    config = pipeline.index_config({**DEFAULT_SETTINGS, "question_index": True})
    with pytest.raises(ValueError, match="LLM"):
        pipeline.build_index(pages, config, llm=None, log=lambda m: None)


# post retrieval -----------------------------------------------------------
def test_corrective_rag_drops_what_the_llm_rejects(ask):
    result = ask("How do plants make oxygen?", llm=FakeLLM(), corrective_rag=True)  # fake keeps source 1
    assert len(result["chunks"]) == 1 and len(result["debug"]["dropped_chunks"]) == 3


def test_compression_adds_the_text_sent_to_the_llm(ask):
    result = ask("How do plants make oxygen?", llm=FakeLLM(), contextual_compression=True)
    assert all(c["compressed_text"] == "COMPRESSED" for c in result["chunks"])


def test_post_retrieval_steps_can_use_their_own_model(ask):
    llm, cheap = FakeLLM(model="answer-model"), FakeLLM(model="cheap-model")
    result = ask("How do plants make oxygen?", llm=llm, post_llm=cheap, corrective_rag=True, contextual_compression=True, query_strategy="Rewrite")
    assert {u["label"] for u in cheap.usage} == {"Corrective RAG", "Contextual compression"}
    assert [u["label"] for u in llm.usage] == ["Query rewrite", "Generation"]
    # the debug panel gets both, each tagged with its model
    by_model = {(u["label"], u["model"]) for u in result["debug"]["llm_usage"]}
    assert ("Corrective RAG", "cheap-model") in by_model and ("Generation", "answer-model") in by_model
    details = {s["name"]: s["detail"] for s in result["debug"]["steps"]}
    assert details["Corrective RAG"].startswith("cheap-model") and details["Generation"] == "answer-model"


def test_post_retrieval_steps_default_to_the_answer_model(ask):
    llm = FakeLLM(model="answer-model")
    result = ask("How do plants make oxygen?", llm=llm, corrective_rag=True)
    assert [u["label"] for u in llm.usage] == ["Corrective RAG", "Generation"]
    assert len(result["debug"]["llm_usage"]) == 2  # not counted twice


def test_positioning_reorders_without_losing_chunks(ask):
    plain = [c["chunk_id"] for c in ask("How do plants make oxygen?")["chunks"]]
    moved = [c["chunk_id"] for c in ask("How do plants make oxygen?", chunk_positioning=True)["chunks"]]
    assert moved == [plain[0], plain[2], plain[3], plain[1]]


def test_llm_steps_are_skipped_without_llm(ask):
    result = ask("How do plants make oxygen?", corrective_rag=True, contextual_compression=True)
    steps = steps_of(result)
    assert not steps["Corrective RAG"] and not steps["Contextual compression"]
    assert len(result["chunks"]) == DEFAULT_SETTINGS["retrieval_n_chunks"]


def test_everything_switched_on_together(ask):
    llm = FakeLLM()
    result = ask(
        "plants and revolution?", llm=llm, chunk_strategy="Hierarchical", retrieval_mode="Hybrid",
        question_index=True, rerank=True, query_strategy="Decompose",
        corrective_rag=True, contextual_compression=True, chunk_positioning=True,
    )
    assert all(steps_of(result).values())
    assert result["answer"]
    assert {u["label"] for u in llm.usage} == {"Query decomposition", "Corrective RAG", "Contextual compression", "Generation"}


def test_an_index_survives_many_newer_ones(pages):
    # Chroma collections went silent once enough newer ones existed, and only when read in between
    def build_and_ask():
        index = pipeline.build_index(pages, pipeline.index_config(DEFAULT_SETTINGS), log=lambda m: None)
        assert pipeline.answer_question("event horizon", index, DEFAULT_SETTINGS, None)["chunks"]
        return index

    indexes = [build_and_ask() for _ in range(40)]
    for index in indexes:
        result = pipeline.answer_question("event horizon", index, DEFAULT_SETTINGS, None)
        assert result["chunks"][0]["page"] == 3


# index config -------------------------------------------------------------
def test_query_time_settings_do_not_change_the_index_config():
    base = pipeline.index_config(DEFAULT_SETTINGS)
    for key, value in {
        "retrieval_n_chunks": 9, "rerank": True, "query_strategy": "HyDE", "hybrid_dense_weight": 0.9,
        "corrective_rag": True, "chunk_positioning": True, "max_tokens": 1,
        "bm25_language": "german",       # not used in Dense mode
        "chunk_parent_size": 999,        # not used by Semantic chunking
    }.items():
        assert pipeline.index_config({**DEFAULT_SETTINGS, key: value}) == base, key


@pytest.mark.parametrize("key, value", [
    ("chunk_strategy", "Fixed size"), ("chunk_size", 300), ("chunk_overlap", 5), ("retrieval_mode", "Hybrid"),
    ("dense_model", "BAAI/bge-small-en-v1.5"), ("question_index", True),
])
def test_index_time_settings_change_the_index_config(key, value):
    assert pipeline.index_config({**DEFAULT_SETTINGS, key: value}) != pipeline.index_config(DEFAULT_SETTINGS)


def test_custom_stopwords_are_split_and_cleaned():
    config = pipeline.index_config({
        **DEFAULT_SETTINGS, "retrieval_mode": "Sparse", "bm25_stopwords_mode": "Custom", "bm25_stopwords_custom": " the, a ,,of ",
    })
    assert config["bm25_stopwords"] == ["the", "a", "of"]


# indexing errors ----------------------------------------------------------
def test_pdf_without_text_is_rejected():
    with pytest.raises(ValueError, match="No text"):
        pipeline.build_index([], pipeline.index_config(DEFAULT_SETTINGS), log=lambda m: None)


def test_openai_embeddings_need_a_key(pages):
    config = pipeline.index_config({**DEFAULT_SETTINGS, "dense_model": "OpenAI Embedding"})
    with pytest.raises(ValueError, match="OpenAI API key"):
        pipeline.build_index(pages, config, openai_api_key="", log=lambda m: None)
