"""Search an index built by pipeline.build_index.

Every function returns chunks as dicts:
  {"chunk_id", "text", "page", "score", "score_type", "parent_text", "parent_id"}
score_type says what the number means: "cosine" (0-1 similarity) or "bm25" (unbounded).
"""


def _to_chunk(chunk_id: str, text: str, metadata: dict, score: float, score_type: str) -> dict:
    return {
        "chunk_id": chunk_id,
        "text": text,
        "page": metadata["page"],
        "score": round(float(score), 4),
        "score_type": score_type,
        "parent_text": metadata["parent_text"] or None,
        "parent_id": metadata["parent_id"] or None,
    }


def search_dense(dense: dict, query: str, k: int) -> list[dict]:
    collection = dense["store"]
    results = collection.query(
        query_embeddings=[dense["indexer"].process_query(query)],
        n_results=min(k, collection.count()),
    )
    return [
        _to_chunk(chunk_id, text, metadata, 1 - distance, "cosine")
        for chunk_id, text, metadata, distance in zip(
            results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]


def search_sparse(sparse: dict, query: str, k: int) -> list[dict]:
    store = sparse["store"]
    results, scores = store["bm25"].retrieve(
        sparse["indexer"].process_query(query),
        k=min(k, len(store["ids"])),  # bm25s raises when k exceeds the corpus size
        show_progress=False,
    )
    return [
        _to_chunk(store["ids"][idx], store["texts"][idx], store["metadatas"][idx], score, "bm25")
        for idx, score in zip(results[0], scores[0])
        if score > 0  # 0 means no query term matched this chunk
    ]


def search(index: dict, query: str, k: int) -> list[dict]:
    if index["dense"]:
        return search_dense(index["dense"], query, k)
    return search_sparse(index["sparse"], query, k)


def merge_results(results_per_query: list[list[dict]], n_chunks: int) -> list[dict]:
    """Merge the result lists of several queries into one list of n_chunks.

    Takes chunks round-robin (best of query 1, best of query 2, ...) so every
    query contributes, and keeps the highest score for chunks found twice.
    """
    best = {}
    for found in results_per_query:
        for chunk in found:
            seen = best.get(chunk["chunk_id"])
            if seen is None or chunk["score"] > seen["score"]:
                best[chunk["chunk_id"]] = chunk

    merged, taken = [], set()
    for rank in range(max((len(f) for f in results_per_query), default=0)):
        for found in results_per_query:
            if rank < len(found) and found[rank]["chunk_id"] not in taken:
                taken.add(found[rank]["chunk_id"])
                merged.append(best[found[rank]["chunk_id"]])
    return merged[:n_chunks]
