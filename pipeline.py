"""The RAG pipeline, independent of the Streamlit UI.

build_index() turns PDF pages into an index, answer_question() runs one
question through it. Both take plain dicts so they can be driven from a
script or a test as well as from app.py.
"""
from ingestion.chunker import chunk_documents
from indexing.indexers import get_dense_indexer, get_sparse_indexer, OPENAI_EMBEDDING
from retrieval.retriever import search, merge_results
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
    return config


def build_index(
    pages: list[dict],
    config: dict,
    session_id: str,
    openai_api_key: str | None = None,
    hf_token: str | None = None,
    log=print,
) -> dict:
    """Chunk the pages and index them as config (from index_config) describes.

    Returns {"config", "chunk_count", "dense", "sparse"}; dense and sparse are
    {"indexer", "store"} or None.
    """
    if not pages:
        raise ValueError("No text could be extracted from this PDF. Scanned PDFs need OCR first.")

    log(f"Chunking {len(pages)} pages ({config['chunk_strategy']})...")
    chunks = chunk_documents(
        pages, config["chunk_strategy"], config["chunk_size"], config["chunk_overlap"],
        config.get("chunk_parent_size"),
    )
    index = {"config": config, "chunk_count": len(chunks), "dense": None, "sparse": None}

    if "dense_model" in config:
        model = config.get("openai_embedding_model_name", config["dense_model"])
        log(f"Embedding {len(chunks)} chunks with {model}...")
        indexer = get_dense_indexer(config, openai_api_key, hf_token)
        vectors = indexer.process_chunks(chunks)
        index["dense"] = {"indexer": indexer, "store": indexer.build_index(chunks, vectors, session_id)}

    if "bm25_language" in config:
        log(f"Building BM25 index over {len(chunks)} chunks...")
        indexer = get_sparse_indexer(config)
        tokens = indexer.process_chunks(chunks)
        index["sparse"] = {"indexer": indexer, "store": indexer.build_index(chunks, tokens, session_id)}

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
    query_result = handle_query(question, strategy, llm)
    queries = query_result["queries"]
    step("Query strategy", strategy != NO_STRATEGY, strategy if strategy != NO_STRATEGY else "raw question")

    n_chunks = settings["retrieval_n_chunks"]
    results_per_query = [search(index, q, n_chunks, settings["hybrid_dense_weight"]) for q in queries]
    query_chunk_map = {q: [c["chunk_id"] for c in found] for q, found in zip(queries, results_per_query)}
    chunks = merge_results(results_per_query, n_chunks)
    step(
        f"{index['config']['retrieval_mode']} retrieval", True,
        f"{len(chunks)} chunks from {len(queries)} quer{'y' if len(queries) == 1 else 'ies'}",
    )

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
            "steps": steps,
            "llm_usage": llm.usage if llm else [],
        },
    }
