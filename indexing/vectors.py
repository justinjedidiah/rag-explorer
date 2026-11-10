from sentence_transformers import SentenceTransformer
from openai import OpenAI
from typing import TypedDict
from indexing.vectorstore import build_index, load_index

class SentenceTransformerIndexerParams(TypedDict):
    model_name: str
    hf_token: str|None
class SentenceTransformerIndexer:
    def __init__(self, model_name: str, hf_token: str|None = None):
        if hf_token:
            self.encoder = SentenceTransformer(model_name, token=hf_token)
        else:
            self.encoder = SentenceTransformer(model_name)

    def process_chunks(self, chunks: list[dict]) -> list[list[float]]:
        texts = [c["text"] for c in chunks]
        vectors = self.encoder.encode(texts, show_progress_bar=True)
        return vectors.tolist()
    
    def build_index(self, chunks: list[dict], vectors: list[list[float]], session_id: str):
        return build_index(chunks, vectors, session_id)
    
    def load_index(self, session_id: str):
        return load_index(session_id)

    def process_query(self, query: str) -> list[float]:
        return self.encoder.encode([query])[0].tolist()

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

class OpenAIIndexerParams(TypedDict):
    model_name: str
    api_key: str
class OpenAIIndexer:
    def __init__(self, model_name: str, api_key:str):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.tokens_used = {}

    def process_chunks(self, chunks: list[dict]) -> list[list[float]]:
        texts = [c["text"] for c in chunks]
        vectors, total_tokens = [], 0
        for batch in _batches(texts):
            response = self.client.embeddings.create(
                input=batch,
                model=self.model_name
            )
            total_tokens += response.usage.total_tokens
            vectors.extend(d.embedding for d in response.data)
        self.tokens_used = {"chunks" : total_tokens}
        return vectors
    
    def build_index(self, chunks: list[dict], vectors: list[list[float]], session_id: str):
        return build_index(chunks, vectors, session_id)
    
    def load_index(self, session_id: str):
        return load_index(session_id)
    
    def process_query(self, query: str) -> list[float]:
        response = self.client.embeddings.create(
            input=[query],
            model=self.model_name
        )
        self.tokens_used[f"{len(self.tokens_used)}. {query}"] = response.usage.total_tokens
        return response.data[0].embedding
