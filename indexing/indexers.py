from indexing.vectors import SentenceTransformerIndexer, SentenceTransformerIndexerParams, OpenAIIndexer, OpenAIIndexerParams
from indexing.nonvectors import BM25Indexer, BM25IndexerParams
import json

IndexerClasses = OpenAIIndexer | SentenceTransformerIndexer | BM25Indexer

_indexers = {}

def get_indexer(
    retrieval_mode,
    model_key,
    params: SentenceTransformerIndexerParams|OpenAIIndexerParams|BM25IndexerParams,
):
    indexer_key = json.dumps(params, sort_keys=True) + model_key
    if indexer_key not in _indexers:
        if retrieval_mode == "Dense":
            if model_key == "OpenAI Embedding":
                _indexers[indexer_key] = OpenAIIndexer(**params)
            else:
                _indexers[indexer_key] = SentenceTransformerIndexer(**params)
        elif retrieval_mode == "Sparse":
            if model_key == "BM25":
                _indexers[indexer_key] = BM25Indexer(**params)
    return _indexers[indexer_key]