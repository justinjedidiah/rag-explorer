from functools import lru_cache
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from indexing.vectorstore import build_index


@lru_cache(maxsize=2)
def _load_encoder(model_name: str, hf_token: str | None) -> SentenceTransformer:
    """Models are large and stateless, so share them between sessions and keep
    at most two in memory."""
    return SentenceTransformer(model_name, token=hf_token or None)


class SentenceTransformerIndexer:
    def __init__(self, model_name: str, hf_token: str | None = None):
        self.encoder = _load_encoder(model_name, hf_token)
        self.usage: list[dict] = []  # local models cost no tokens

    def process_chunks(self, chunks: list[dict]) -> list[list[float]]:
        texts = [c["text"] for c in chunks]
        return self.encoder.encode(texts, show_progress_bar=False).tolist()

    def build_index(self, chunks: list[dict], vectors: list[list[float]]) -> dict:
        return build_index(chunks, vectors)

    def process_query(self, query: str) -> list[float]:
        return self.encoder.encode([query], show_progress_bar=False)[0].tolist()


def _batches(texts: list[str], max_items: int = 256, max_chars: int = 400_000):
    """Split texts so one embeddings request stays under OpenAI's per-request
    limits (2048 inputs, ~300k tokens). 400k chars is roughly 100k tokens."""
    batch, chars = [], 0
    for text in texts:
        if batch and (len(batch) >= max_items or chars + len(text) > max_chars):
            yield batch
            batch, chars = [], 0
        batch.append(text)
        chars += len(text)
    if batch:
        yield batch


class OpenAIIndexer:
    """One instance per indexed document, so .usage is that document's embedding cost."""

    def __init__(self, model_name: str, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.usage: list[dict] = []  # [{"label", "tokens"}]

    def process_chunks(self, chunks: list[dict]) -> list[list[float]]:
        texts = [c["text"] for c in chunks]
        vectors, total_tokens = [], 0
        for batch in _batches(texts):
            response = self.client.embeddings.create(input=batch, model=self.model_name)
            total_tokens += response.usage.total_tokens
            vectors.extend(d.embedding for d in response.data)
        self.usage.append({"label": f"Indexing {len(texts)} chunks", "tokens": total_tokens})
        return vectors

    def build_index(self, chunks: list[dict], vectors: list[list[float]]) -> dict:
        return build_index(chunks, vectors)

    def process_query(self, query: str) -> list[float]:
        response = self.client.embeddings.create(input=[query], model=self.model_name)
        self.usage.append({"label": f"Query: {query[:60]}", "tokens": response.usage.total_tokens})
        return response.data[0].embedding
