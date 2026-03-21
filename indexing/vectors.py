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
        response = self.client.embeddings.create(
            input=texts,
            model=self.model_name
        )
        self.tokens_used = {"chunks" : response.usage.total_tokens}
        return [d.embedding for d in response.data]
    
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
