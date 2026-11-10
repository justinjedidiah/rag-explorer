"""The RAG pipeline, independent of the Streamlit UI.

build_index() turns PDF pages into an index, answer_question() runs one
question through it. Both take plain dicts so they can be driven from a
script or a test as well as from app.py.
"""
from ingestion.chunker import chunk_documents
from indexing.questions import generate_question_entries
from indexing.indexers import get_dense_indexer, get_sparse_indexer, OPENAI_EMBEDDING
from retrieval.retriever import search, merge_results
from retrieval.reranker import rerank
from retrieval.postprocess import corrective_filter, compress, reposition
from query.handler import handle_query, NO_STRATEGY
from generation.generator import LLM, generate

# Which retrieval modes need which kind of index
DENSE_MODES = ("Dense", "Hybrid")
SPARSE_MODES = ("Sparse", "Hybrid")


def index_config(settings: dict) -> dict:
    """The part of the settings an index is built from.

    Stored on the index; when the sidebar no longer produces the same config,
    the document has to be re-indexed for the change to take effect.
    """
    config = {
        "chunk_strategy": settings["chunk_strategy"],
        "chunk_size": settings["chunk_size"],
        "chunk_overlap": settings["chunk_overlap"],
        "retrieval_mode": settings["retrieval_mode"],
    }
    if settings["chunk_strategy"] == "Hierarchical":
        config["chunk_parent_size"] = settings["chunk_parent_size"]

    if settings["retrieval_mode"] in DENSE_MODES:
        config["dense_model"] = settings["dense_model"]
        if settings["dense_model"] == OPENAI_EMBEDDING:
            config["openai_embedding_model_name"] = settings["openai_embedding_model_name"]

    if settings["retrieval_mode"] in SPARSE_MODES:
        config["bm25_language"] = settings["bm25_language"]
        config["bm25_stemming"] = settings["bm25_stemming"]
        if settings["bm25_stopwords_mode"] == "Language default":
            config["bm25_stopwords"] = settings["bm25_language"]
        elif settings["bm25_stopwords_mode"] == "Custom":
            words = [w.strip() for w in settings["bm25_stopwords_custom"].split(",")]
            config["bm25_stopwords"] = [w for w in words if w]
        else:
            config["bm25_stopwords"] = None

    if settings["question_index"]:
        config["question_index"] = True
        config["questions_per_chunk"] = settings["questions_per_chunk"]
    return config


def build_index(
    pages: list[dict],
    config: dict,
    session_id: str,
    openai_api_key: str | None = None,
    hf_token: str | None = None,
    llm: LLM | None = None,
    log=print,
    progress=None,
) -> dict:
    """Chunk the pages and index them as config (from index_config) describes.

    llm is only needed for a question index; progress(fraction) is called while
    its questions are generated.
    Returns {"config", "chunk_count", "dense", "sparse", "llm_usage"}; dense and
    sparse are {"indexer", "store", "entries_per_chunk"} or None.
    """
    if not pages:
        raise ValueError("No text could be extracted from this PDF. Scanned PDFs need OCR first.")

    log(f"Chunking {len(pages)} pages ({config['chunk_strategy']})...")
    chunks = chunk_documents(
        pages, config["chunk_strategy"], config["chunk_size"], config["chunk_overlap"],
        config.get("chunk_parent_size"),
    )
    index = {"config": config, "chunk_count": len(chunks), "dense": None, "sparse": None, "llm_usage": []}

    # What gets embedded / tokenised: the chunks, or generated questions pointing back to them
    entries, entries_per_chunk = chunks, 1
    if config.get("question_index"):
        if llm is None:
            raise ValueError("Indexing hypothetical questions needs the LLM enabled.")
        entries_per_chunk = config["questions_per_chunk"]
        log(f"Writing {entries_per_chunk} questions for each of {len(chunks)} chunks with {llm.model}...")
        entries = generate_question_entries(chunks, llm, entries_per_chunk, progress)
        index["llm_usage"] = llm.usage

    if "dense_model" in config:
        model = config.get("openai_embedding_model_name", config["dense_model"])
        log(f"Embedding {len(entries)} texts with {model}...")
        indexer = get_dense_indexer(config, openai_api_key, hf_token)
        vectors = indexer.process_chunks(entries)
        store = indexer.build_index(entries, vectors, session_id)
        index["dense"] = {"indexer": indexer, "store": store, "entries_per_chunk": entries_per_chunk}

    if "bm25_language" in config:
        log(f"Building BM25 index over {len(entries)} texts...")
        indexer = get_sparse_indexer(config)
        tokens = indexer.process_chunks(entries)
        store = indexer.build_index(entries, tokens, session_id)
        index["sparse"] = {"indexer": indexer, "store": store, "entries_per_chunk": entries_per_chunk}

    return index


def answer_question(question: str, index: dict, settings: dict, llm: LLM | None) -> dict:
    """Run one question through the pipeline.

    llm=None means no LLM calls at all: the raw question is used for retrieval
    and no answer is generated.
    Returns {"answer", "chunks", "debug"}; debug["steps"] lists what ran.
    """
    steps = []

    def step(name: str, ran: bool, detail: str = ""):
        steps.append({"name": name, "ran": ran, "detail": detail})

    strategy = settings["query_strategy"] if llm else NO_STRATEGY
    strategy_detail = strategy if strategy != NO_STRATEGY else "raw question"
    if strategy == "HyDE" and index["config"].get("question_index"):
        # the index holds questions, so the query must stay a question
        strategy, strategy_detail = NO_STRATEGY, "HyDE skipped: the index stores questions, not answers"
    query_result = handle_query(question, strategy, llm)
    queries = query_result["queries"]
    step("Query strategy", strategy != NO_STRATEGY, strategy_detail)

    n_chunks = settings["retrieval_n_chunks"]
    # with reranking, retrieve a wide candidate set cheaply and let the cross-encoder pick
    n_retrieve = max(settings["rerank_candidates"], n_chunks) if settings["rerank"] else n_chunks
    results_per_query = [search(index, q, n_retrieve, settings["hybrid_dense_weight"]) for q in queries]
    query_chunk_map = {q: [c["chunk_id"] for c in found] for q, found in zip(queries, results_per_query)}
    chunks = merge_results(results_per_query, n_retrieve)
    step(
        f"{index['config']['retrieval_mode']} retrieval", True,
        f"{len(chunks)} chunks from {len(queries)} quer{'y' if len(queries) == 1 else 'ies'}",
    )

    if settings["rerank"]:
        n_candidates = len(chunks)
        chunks = rerank(question, chunks, n_chunks)  # always against the user's own question
        step("Reranker", True, f"kept {len(chunks)} of {n_candidates} candidates")
    else:
        step("Reranker", False, "disabled")

    dropped = []
    if settings["corrective_rag"] and llm:
        chunks, dropped = corrective_filter(question, chunks, llm)
        step("Corrective RAG", True, f"kept {len(chunks)}, dropped {len(dropped)} as irrelevant")
    else:
        step("Corrective RAG", False, "needs the LLM" if settings["corrective_rag"] else "disabled")

    if settings["contextual_compression"] and llm:
        chunks = compress(question, chunks, llm)
        n_compressed = sum(1 for c in chunks if c.get("compressed_text"))
        step("Contextual compression", True, f"compressed {n_compressed} of {len(chunks)} chunks")
    else:
        step("Contextual compression", False, "needs the LLM" if settings["contextual_compression"] else "disabled")

    if settings["chunk_positioning"]:
        chunks = reposition(chunks)
        step("Chunk positioning", True, "best chunks first and last, weakest in the middle")
    else:
        step("Chunk positioning", False, "disabled")

    answer = None
    if llm:
        answer = generate(question, chunks, llm, settings["max_tokens"])
    step("Generation", llm is not None, llm.model if llm else "disabled or no API key")

    return {
        "answer": answer,
        "chunks": chunks,
        "debug": {
            "question": question,
            "query_strategy": strategy,
            "query_display": query_result["display"],
            "query_chunk_map": query_chunk_map,
            "chunks": chunks,
            "dropped_chunks": dropped,
            "steps": steps,
            "llm_usage": llm.usage if llm else [],
        },
    }
