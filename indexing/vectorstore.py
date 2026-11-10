import re
import time
import chromadb

# In memory on purpose. An index only lives as long as its browser session, so
# nothing needs to survive a restart, and PersistentClient loses small
# collections: below its sync threshold their HNSW index exists only in a cache,
# and once about nine collections exist the oldest are evicted and every query
# on them fails with "Error creating hnsw segment reader: Nothing found on disk".
client = chromadb.EphemeralClient()


def chunk_metadata(chunk: dict) -> dict:
    """What is stored next to a chunk's text in every index (Chroma has no None values)."""
    return {
        "page": chunk["page"],
        "parent_text": chunk.get("parent_text") or "",
        "parent_id": chunk.get("parent_id") or "",
        # set when the indexed text is a generated question that stands in for the chunk
        "source_chunk_id": chunk.get("source_chunk_id") or "",
        "source_text": chunk.get("source_text") or "",
    }


def get_session_collection_name(session_id: str) -> str:
    """Make Chroma collection name based on streamlit session id"""
    # Chroma only allows [a-zA-Z0-9._-] in collection names
    return "docs_" + re.sub(r"[^a-zA-Z0-9._-]", "-", session_id)


def build_index(chunks: list[dict], vectors: list[list[float]], session_id: str):
    """Make Chroma collection, delete old collection if already exist"""
    cleanup_old_collections()
    name = get_session_collection_name(session_id)
    existing = [c.name for c in client.list_collections()]
    if name in existing:
        client.delete_collection(name)

    collection = client.create_collection(
        name,
        metadata={
            "hnsw:space": "cosine",
            "created_at": time.time(),
        }
    )
    # Chroma rejects a single add() above its max batch size
    batch_size = client.get_max_batch_size()
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        collection.add(
            ids=[c["chunk_id"] for c in batch],
            documents=[c["text"] for c in batch],
            embeddings=vectors[start:start + batch_size],
            metadatas=[chunk_metadata(c) for c in batch],
        )
    return collection


def cleanup_old_collections(max_age_hours: int = 24):
    """Free the collections of sessions that are long gone. A session's
    collection cannot be reached once the session ends, so without this a
    long-running server would keep every document ever indexed in memory."""
    now = time.time()
    for col in client.list_collections():
        created_at = (col.metadata or {}).get("created_at", 0)
        if (now - created_at) / 3600 > max_age_hours:
            try:
                client.delete_collection(col.name)
            except Exception:
                pass
