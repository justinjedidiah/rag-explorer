from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_fixed(pages: list[dict], size: int, overlap: int) -> list[dict]:
    chunks = []
    for page in pages:
        text = page["text"]
        start = 0
        i = 0
        while start < len(text):
            chunk_text = text[start:start + size]
            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text,
                    "page": page["page"],
                    "chunk_id": f"p{page['page']}_c{i}",
                    "parent_text": None,
                    "parent_id": None,
                })
            start += size - overlap
            i += 1
    return chunks

def chunk_semantic(pages: list[dict], size: int, overlap: int) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""]  # tries paragraph first, then sentence, etc
    )
    chunks = []
    for page in pages:
        texts = splitter.split_text(page["text"])
        for i, text in enumerate(texts):
            chunks.append({
                "text": text,
                "page": page["page"],
                "chunk_id": f"p{page['page']}_c{i}",
                "parent_text": None,
                "parent_id": None,
            })
    return chunks


def chunk_hierarchical(pages: list[dict], size: int, overlap: int, parent_size: int) -> list[dict]:
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = []
    for page in pages:
        parents = parent_splitter.split_text(page["text"])
        chunk_num = 0
        for p_idx, parent_text in enumerate(parents):
            parent_id = f"p{page['page']}_parent{p_idx}"
            children = child_splitter.split_text(parent_text)
            for child_text in children:
                chunks.append({
                    "text": child_text,
                    "page": page["page"],
                    "chunk_id": f"p{page['page']}_c{chunk_num}",
                    "parent_text": parent_text,
                    "parent_id": parent_id,
                })
                chunk_num += 1
    return chunks

def chunk_documents(pages, strategy, size, overlap, parent_size=None):
    if strategy == "Fixed size":
        return chunk_fixed(pages, size, overlap)
    elif strategy == "Semantic":
        return chunk_semantic(pages, size, overlap)
    elif strategy == "Hierarchical":
        return chunk_hierarchical(pages, size, overlap, parent_size)