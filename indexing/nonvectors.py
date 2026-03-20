import bm25s
import Stemmer
from typing import TypedDict

STOPWORDS_LANG_BM25 = ["english", "en","english_plus", "en_plus","german", "de","dutch", "nl","french", "fr","spanish", "es","portuguese", "pt","italian", "it","russian", "ru","swedish", "sv","norwegian", "no","chinese", "zh","turkish", "tr"]

class BM25IndexerParams(TypedDict):
    language: str
    stemming: bool
    stopwords: list|str

class BM25SModdedCollection:
    """Adding chunk texts and metadata because bm25s.BM25 only store the indices"""
    ids: list[str]
    collection: bm25s.BM25
    chunk_texts: list[str]
    metadatas: list[dict]
class BM25Indexer:
    def __init__(self, language: str, stemming: bool, stopwords: list|str):
        if stemming and language in Stemmer.algorithms():
            self.stemmer = Stemmer.Stemmer(language)
        else:
            self.stemmer = None
        self.stopwords = stopwords
    
    def process_chunks(self, chunks: list[dict]) -> bm25s.tokenization.Tokenized:
        corpus = [c["text"] for c in chunks]
        corpus_tokens = bm25s.tokenize(corpus, stopwords=self.stopwords, stemmer=self.stemmer)
        return corpus_tokens
    
    def build_index(self, chunks: list[dict], corpus_tokens: bm25s.tokenization.Tokenized) -> BM25SModdedCollection:
        collection = bm25s.BM25()
        collection.index(corpus_tokens)
        chunk_ids = [c["chunk_id"] for c in chunks]
        chunk_texts = [c["text"] for c in chunks]
        metadatas=[{
            "page": c["page"],
            "parent_text": c.get("parent_text") or "",
            "parent_id": c.get("parent_id") or "",
        } for c in chunks]
        modded_collection = {"ids": chunk_ids, "collection": collection, "chunk_texts": chunk_texts, "metadatas": metadatas}
        return modded_collection

    def load_index(self, session_id: str) -> None:
        return None

    def process_query(self, query: str) -> bm25s.tokenization.Tokenized:
        return bm25s.tokenize(query, stopwords=self.stopwords, stemmer=self.stemmer)