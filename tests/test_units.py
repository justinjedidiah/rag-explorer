"""Pure functions: no models, no index."""
import pytest

from generation.generator import parse_json_list, build_prompt, chunk_text_for_llm
from generation.models import is_openai_chat_model, default_model
from indexing.nonvectors import BM25Indexer
from indexing.vectors import _batches
from ingestion.chunker import chunk_documents
from retrieval.postprocess import corrective_filter, reposition
from retrieval.retriever import merge_results, search_sparse

PAGES = [{"text": "First sentence here. " * 30, "page": 1}, {"text": "Another page of text. " * 30, "page": 2}]


# chunking -----------------------------------------------------------------
@pytest.mark.parametrize("strategy", ["Fixed size", "Semantic", "Hierarchical"])
def test_chunks_respect_size_and_have_unique_ids(strategy):
    chunks = chunk_documents(PAGES, strategy, size=100, overlap=10, parent_size=300)
    assert chunks
    assert all(len(c["text"]) <= 100 for c in chunks)
    assert len({c["chunk_id"] for c in chunks}) == len(chunks)
    assert {c["page"] for c in chunks} == {1, 2}


def test_hierarchical_children_point_to_their_parent():
    chunks = chunk_documents(PAGES, "Hierarchical", size=100, overlap=10, parent_size=300)
    assert all(c["text"] in c["parent_text"] and len(c["parent_text"]) <= 300 for c in chunks)


@pytest.mark.parametrize("overlap", [100, 150])
def test_overlap_not_smaller_than_size_is_rejected(overlap):
    # used to loop forever in chunk_fixed
    with pytest.raises(ValueError, match="overlap"):
        chunk_documents(PAGES, "Fixed size", size=100, overlap=overlap)


def test_parent_not_larger_than_chunk_is_rejected():
    with pytest.raises(ValueError, match="Parent size"):
        chunk_documents(PAGES, "Hierarchical", size=100, overlap=10, parent_size=100)


# merging and ordering -----------------------------------------------------
def _chunk(chunk_id, score):
    return {"chunk_id": chunk_id, "score": score}


def test_merge_takes_from_every_query():
    # the old merge returned only the first query's results
    first = [_chunk("a1", 0.9), _chunk("a2", 0.8), _chunk("a3", 0.7)]
    second = [_chunk("b1", 0.6), _chunk("b2", 0.5), _chunk("b3", 0.4)]
    merged = merge_results([first, second], n_chunks=4)
    assert [c["chunk_id"] for c in merged] == ["a1", "b1", "a2", "b2"]


def test_merge_keeps_best_score_of_a_duplicate():
    merged = merge_results([[_chunk("x", 0.5)], [_chunk("x", 0.9)]], n_chunks=5)
    assert merged == [_chunk("x", 0.9)]


def test_merge_of_nothing():
    assert merge_results([], 5) == [] and merge_results([[], []], 5) == []


@pytest.mark.parametrize("best_first, expected", [
    ([1, 2, 3, 4, 5], [1, 3, 5, 4, 2]),
    ([1, 2], [1, 2]),
    ([1], [1]),
    ([], []),
])
def test_reposition_puts_best_first_and_last(best_first, expected):
    assert reposition(best_first) == expected


# LLM reply parsing --------------------------------------------------------
@pytest.mark.parametrize("raw, expected", [
    ('["a", "b"]', ["a", "b"]),
    ('Here you go:\n["a", "b"]\nHope that helps', ["a", "b"]),
    ("no list here", []),
    ("[not json]", []),
    ("", []),
])
def test_parse_json_list(raw, expected):
    assert parse_json_list(raw) == expected


class ReplyLLM:
    def __init__(self, reply):
        self.reply = reply

    def complete(self, prompt, max_tokens=2000, label=""):
        return self.reply


THREE = [{"text": t, "page": 1} for t in ("one", "two", "three")]


def test_corrective_filter_splits_kept_and_dropped():
    kept, dropped = corrective_filter("q", THREE, ReplyLLM("Relevant: [1, 3]"))
    assert [c["text"] for c in kept] == ["one", "three"] and [c["text"] for c in dropped] == ["two"]


def test_corrective_filter_can_drop_everything():
    kept, dropped = corrective_filter("q", THREE, ReplyLLM("[]"))
    assert kept == [] and len(dropped) == 3


def test_corrective_filter_keeps_all_when_reply_is_unreadable():
    kept, dropped = corrective_filter("q", THREE, ReplyLLM("I think the first one"))
    assert kept == THREE and dropped == []


# what the LLM is sent -----------------------------------------------------
def test_llm_text_prefers_compressed_then_parent_then_chunk():
    assert chunk_text_for_llm({"text": "c"}) == "c"
    assert chunk_text_for_llm({"text": "c", "parent_text": "p"}) == "p"
    assert chunk_text_for_llm({"text": "c", "parent_text": "p", "compressed_text": "z"}) == "z"


def test_prompt_numbers_sources_with_pages():
    prompt = build_prompt("Why?", [{"text": "alpha", "page": 3}, {"text": "beta", "page": 7}])
    assert "[Source 1 | Page 3]\nalpha" in prompt and "[Source 2 | Page 7]\nbeta" in prompt
    assert "Question: Why?" in prompt


# embeddings batching ------------------------------------------------------
def test_batches_keep_order_and_respect_limits():
    texts = [f"{i}" * 100 for i in range(1000)]
    batches = list(_batches(texts, max_items=256, max_chars=20_000))
    assert [t for batch in batches for t in batch] == texts
    assert all(len(b) <= 256 and sum(map(len, b)) <= 20_000 for b in batches)


def test_one_oversized_text_still_gets_its_own_batch():
    assert list(_batches(["x" * 50, "y" * 5000, "z" * 50], max_chars=1000)) == [["x" * 50], ["y" * 5000], ["z" * 50]]


# BM25 ---------------------------------------------------------------------
def _bm25(texts, **kwargs):
    indexer = BM25Indexer(**{"language": "english", "stemming": False, "stopwords": None, **kwargs})
    chunks = [{"chunk_id": f"c{i}", "text": t, "page": 1} for i, t in enumerate(texts)]
    store = indexer.build_index(chunks, indexer.process_chunks(chunks))
    return {"indexer": indexer, "store": store, "entries_per_chunk": 1}


TEXTS = ["the cats are running fast", "dogs bark loudly at night", "a bird sings"]


def test_bm25_stemming_matches_other_word_forms():
    assert [c["text"] for c in search_sparse(_bm25(TEXTS, stemming=True), "run cat", 5)] == [TEXTS[0]]
    assert search_sparse(_bm25(TEXTS, stemming=False), "run cat", 5) == []


def test_bm25_k_larger_than_corpus_and_no_match():
    sparse = _bm25(TEXTS)
    assert len(search_sparse(sparse, "dogs", 50)) == 1  # bm25s raises on k > corpus size
    assert search_sparse(sparse, "zebra", 5) == []


def test_bm25_stemmer_missing_for_language():
    with pytest.raises(ValueError, match="No stemmer"):
        BM25Indexer("klingon", stemming=True, stopwords=None)


# model catalogue ----------------------------------------------------------
@pytest.mark.parametrize("model_id", ["gpt-4o", "gpt-4o-mini", "gpt-5.4-nano", "gpt-6.1-sol", "o3"])
def test_openai_chat_models_are_recognised(model_id):
    assert is_openai_chat_model(model_id)


@pytest.mark.parametrize("model_id", [
    "text-embedding-3-small", "whisper-1", "tts-1-hd", "omni-moderation-latest",
    "gpt-image-2", "gpt-4o-mini-tts", "gpt-4o-transcribe", "gpt-realtime-2", "gpt-audio-mini",
    "gpt-5-search-api", "gpt-5.3-codex", "gpt-5.5-pro", "gpt-4o-2024-08-06", "gpt-live-1",
])
def test_openai_non_chat_models_are_filtered_out(model_id):
    assert not is_openai_chat_model(model_id)


def test_default_model_prefers_the_fallback_choice():
    assert default_model(["new-1", "cheap", "new-2"], fallback=["cheap", "big"]) == "cheap"
    assert default_model(["new-1", "new-2"], fallback=["cheap", "big"]) == "new-1"
