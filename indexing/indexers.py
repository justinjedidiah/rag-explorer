from indexing.vectors import SentenceTransformerIndexer, OpenAIIndexer
from indexing.nonvectors import BM25Indexer

OPENAI_EMBEDDING = "OpenAI Embedding"


def get_dense_indexer(config: dict, openai_api_key: str | None = None, hf_token: str | None = None):
    if config["dense_model"] == OPENAI_EMBEDDING:
        if not openai_api_key:
            raise ValueError("OpenAI embeddings need an OpenAI API key (sidebar → API keys).")
        return OpenAIIndexer(config["openai_embedding_model_name"], openai_api_key)
    return SentenceTransformerIndexer(config["dense_model"], hf_token)


def get_sparse_indexer(config: dict):
    return BM25Indexer(config["bm25_language"], config["bm25_stemming"], config["bm25_stopwords"])
