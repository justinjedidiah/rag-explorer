import bm25s
import Stemmer
from indexing.vectorstore import chunk_metadata

# Languages that have both a bm25s stopword list and a PyStemmer stemmer
BM25_LANGUAGES = ["english", "german", "dutch", "french", "spanish", "portuguese", "italian", "russian", "swedish", "norwegian", "turkish"]


class BM25Indexer:
    usage: list[dict] = []  # BM25 costs no tokens; kept so every indexer exposes .usage

    def __init__(self, language: str, stemming: bool, stopwords: list | str | None):
        self.stemmer = None
        if stemming:
            if language not in Stemmer.algorithms():
                raise ValueError(f"No stemmer available for '{language}'. Turn stemming off or pick another language.")
            self.stemmer = Stemmer.Stemmer(language)
        self.stopwords = stopwords

    def _tokenize(self, texts):
        return bm25s.tokenize(texts, stopwords=self.stopwords, stemmer=self.stemmer, show_progress=False)

    def process_chunks(self, chunks: list[dict]):
        return self._tokenize([c["text"] for c in chunks])

    def build_index(self, chunks: list[dict], corpus_tokens, session_id: str | None = None) -> dict:
        """bm25s.BM25 only stores token indices, so keep ids, texts and metadata next to it."""
        bm25 = bm25s.BM25()
        bm25.index(corpus_tokens, show_progress=False)
        return {
            "bm25": bm25,
            "ids": [c["chunk_id"] for c in chunks],
            "texts": [c["text"] for c in chunks],
            "metadatas": [chunk_metadata(c) for c in chunks],
        }

    def process_query(self, query: str):
        return self._tokenize(query)
