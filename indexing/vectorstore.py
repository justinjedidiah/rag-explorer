"""The dense index: every chunk's embedding in one NumPy matrix, searched exactly.

An index holds one document for one browser session, so it is small enough to
compare a question against every chunk (one matrix product) instead of using an
approximate index. The store is a plain dict that lives in the session's state
and is freed with it; nothing is shared between sessions.

This replaced Chroma, whose in-memory collections could silently return no
results once other collections had been created in the same process.
"""
import numpy as np


def chunk_metadata(chunk: dict) -> dict:
    """What is stored next to a chunk's text in every index."""
    return {
        "page": chunk["page"],
        "page_end": chunk.get("page_end") or chunk["page"],
        "parent_text": chunk.get("parent_text") or "",
        "parent_id": chunk.get("parent_id") or "",
        # set when the indexed text is a generated question that stands in for the chunk
        "source_chunk_id": chunk.get("source_chunk_id") or "",
        "source_text": chunk.get("source_text") or "",
    }


def _unit(vectors) -> np.ndarray:
    """Rows scaled to length 1, so a dot product between two of them is their cosine similarity."""
    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.where(norms == 0, 1, norms)


def build_index(chunks: list[dict], vectors: list[list[float]]) -> dict:
    return {
        "vectors": _unit(vectors),
        "ids": [c["chunk_id"] for c in chunks],
        "texts": [c["text"] for c in chunks],
        "metadatas": [chunk_metadata(c) for c in chunks],
    }


def search(store: dict, query_vector: list[float], k: int) -> list[tuple[int, float]]:
    """(position in the store, cosine similarity) of the k entries closest to the query, best first."""
    scores = store["vectors"] @ _unit(query_vector)
    best = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in best]
