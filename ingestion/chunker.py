"""Split the pages of a document into chunks.

The pages are joined into one text before splitting, so a sentence or paragraph
that runs over a page break stays in one chunk. Every chunk records the page it
starts on ("page") and the page it ends on ("page_end").
"""
from bisect import bisect_right
from langchain_text_splitters import RecursiveCharacterTextSplitter

SEPARATORS = ["\n\n", "\n", ". ", " ", ""]  # tries paragraph first, then sentence, etc


class _Document:
    """All pages as one text, with a way back from a character position to its page."""

    def __init__(self, pages: list[dict]):
        # a page break is joined like a line break inside a page, so splitters treat it the same
        self.text = "\n".join(page["text"] for page in pages)
        self._numbers = [page["page"] for page in pages]
        self._starts, position = [], 0
        for page in pages:
            self._starts.append(position)
            position += len(page["text"]) + 1

    def page_at(self, position: int) -> int:
        return self._numbers[max(bisect_right(self._starts, position) - 1, 0)]

    def chunk(self, number: int, text: str, start: int, parent_text: str | None = None, parent_id: str | None = None) -> dict:
        # measured on the visible text: a chunk that begins with the line break before a page starts on that page
        first = start + len(text) - len(text.lstrip())
        page = self.page_at(first)
        return {
            "text": text,
            "page": page,
            "page_end": self.page_at(max(start + len(text.rstrip()) - 1, first)),
            "chunk_id": f"p{page}_c{number}",
            "parent_text": parent_text,
            "parent_id": parent_id,
        }


def _split(text: str, size: int, overlap: int) -> list[tuple[str, int]]:
    """(piece, position of the piece in text) for each piece the recursive splitter makes."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap, separators=SEPARATORS, add_start_index=True,
    )
    return [(piece.page_content, max(piece.metadata["start_index"], 0)) for piece in splitter.create_documents([text])]


def page_label(chunk: dict) -> str:
    """'page 3', or 'pages 3-4' for a chunk that runs over a page break."""
    page, page_end = chunk["page"], chunk.get("page_end") or chunk["page"]
    return f"page {page}" if page_end == page else f"pages {page}-{page_end}"


def chunk_fixed(pages: list[dict], size: int, overlap: int) -> list[dict]:
    document = _Document(pages)
    chunks = []
    for start in range(0, len(document.text), size - overlap):
        text = document.text[start:start + size]
        if text.strip():
            chunks.append(document.chunk(len(chunks), text, start))
    return chunks


def chunk_semantic(pages: list[dict], size: int, overlap: int) -> list[dict]:
    document = _Document(pages)
    return [document.chunk(i, text, start) for i, (text, start) in enumerate(_split(document.text, size, overlap))]


def chunk_hierarchical(pages: list[dict], size: int, overlap: int, parent_size: int) -> list[dict]:
    document = _Document(pages)
    chunks = []
    parents = _split(document.text, parent_size, min(overlap, parent_size - 1))
    for parent_number, (parent_text, parent_start) in enumerate(parents):
        for text, start in _split(parent_text, size, overlap):
            chunks.append(document.chunk(len(chunks), text, parent_start + start, parent_text, f"parent{parent_number}"))
    return chunks


def chunk_documents(pages, strategy, size, overlap, parent_size=None):
    if overlap >= size:
        # chunk_fixed would never advance, and the splitters reject it anyway
        raise ValueError("Chunk overlap must be smaller than chunk size.")
    if strategy == "Hierarchical" and (parent_size is None or parent_size <= size):
        raise ValueError("Parent size must be larger than chunk size.")
    if strategy == "Fixed size":
        return chunk_fixed(pages, size, overlap)
    elif strategy == "Semantic":
        return chunk_semantic(pages, size, overlap)
    elif strategy == "Hierarchical":
        return chunk_hierarchical(pages, size, overlap, parent_size)
