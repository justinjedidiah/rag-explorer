import time
import chromadb
from config import CHROMA_PATH

client = chromadb.PersistentClient(path=CHROMA_PATH)

def get_session_collection_name(session_id: str) -> str:
    """Make Chroma collection name based on streamlit session id"""
    return f"docs_{session_id[:8]}"

def build_index(chunks: list[dict], vectors: list[list[float]], session_id: str):
    """Make Chroma collection, delete old collection if already exist"""
    name = get_session_collection_name(session_id)
    existing = [c.name for c in client.list_collections()]
    if name in existing:
        client.delete_collection(name)

    collection = client.create_collection(
        name,
        metadata={
            "hnsw:space": "cosine",
            "created_at": time.time(),
            "session_id": session_id,
        }
    )
    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=vectors,
        metadatas=[{
            "page": c["page"],
            "parent_text": c.get("parent_text") or "",
            "parent_id": c.get("parent_id") or "",
        } for c in chunks],
    )
    return collection

def load_index(session_id: str):
    name = get_session_collection_name(session_id)
    existing = [c.name for c in client.list_collections()]
    if name in existing:
        return client.get_collection(name)
    return None

def cleanup_old_collections(max_age_hours: int = 24):
    """Delete collections older than max_age_hours. Call on app startup."""
    now = time.time()
    for col in client.list_collections():
        meta = col.metadata or {}
        created_at = meta.get("created_at", 0)
        age_hours = (now - created_at) / 3600
        if age_hours > max_age_hours:
            try:
                client.delete_collection(col.name)
            except Exception:
                pass