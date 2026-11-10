from indexing.indexers import IndexerClasses
from indexing.nonvectors import BM25SModdedCollection

def retrieve(
    question: str,
    collection,
    n_chunks: int,
    indexer: IndexerClasses,
    retrieval_model: str,
) -> list[dict]:
    if retrieval_model == "BM25":
        chunks = retrieveBM25SCollection(question,collection,n_chunks,indexer)
    else:
        chunks = retrieveChromaCollection(question,collection,n_chunks,indexer)
    return chunks

def retrieveChromaCollection(
    question: str,
    collection,
    n_chunks: int,
    indexer: IndexerClasses,
) -> list[dict]:
    q_vec = indexer.process_query(question)
    results = collection.query(
        query_embeddings=q_vec,
        n_results=n_chunks,
    )

    chunks = []
    for id, text, metadata, distance in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "chunk_id": id,
            "text": text,
            "page": metadata["page"],
            "score": round(1 - distance, 4),
            "parent_text": metadata["parent_text"] or None,
            "parent_id": metadata["parent_id"] or None,
        })
    return chunks

def retrieveBM25SCollection(
    question: str,
    modded_collection: BM25SModdedCollection,
    n_chunks: int,
    indexer: IndexerClasses,
) -> list[dict]:
    q_tokens = indexer.process_query(question)

    collection = modded_collection["collection"]
    chunk_ids = modded_collection["ids"]
    metadatas = modded_collection["metadatas"]
    chunk_texts = modded_collection["chunk_texts"]

    results, scores = collection.retrieve(
        query_tokens=q_tokens,
        k=n_chunks,
    )

    chunks = []
    for idx, score in zip(results[0], scores[0]):
        chunks.append({
            "chunk_id": chunk_ids[idx],
            "text": chunk_texts[idx],
            "page": metadatas[idx]["page"],
            "score": float(score),
            "parent_text": metadatas[idx]["parent_text"] or None,
            "parent_id": metadatas[idx]["parent_id"] or None,
        })
    return chunks

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
