"""Steps that run on the retrieved chunks before they are sent to the LLM."""
import re
from generation.generator import LLM, chunk_text_for_llm


def corrective_filter(question: str, chunks: list[dict], llm: LLM) -> tuple[list[dict], list[dict]]:
    """Corrective RAG: let the LLM judge which chunks are relevant to the question.

    Returns (kept, dropped). One LLM call for all chunks. If the reply cannot be
    read, every chunk is kept.
    """
    if not chunks:
        return [], []
    sources = "\n\n".join(f"[Source {i}]\n{chunk_text_for_llm(c)}" for i, c in enumerate(chunks, 1))
    prompt = f"""Decide which sources contain information that helps answer the question.
Return ONLY a JSON array with the numbers of the helpful sources, e.g. [1, 3]. Return [] if none help.

Question: {question}

{sources}

Helpful sources:"""
    raw = llm.complete(prompt, label="Corrective RAG")
    match = re.search(r"\[[\d,\s]*\]", raw)
    if not match:
        return chunks, []
    relevant = {int(n) for n in re.findall(r"\d+", match.group())}
    kept = [c for i, c in enumerate(chunks, 1) if i in relevant]
    dropped = [c for i, c in enumerate(chunks, 1) if i not in relevant]
    return kept, dropped


def compress(question: str, chunks: list[dict], llm: LLM) -> list[dict]:
    """Contextual compression: cut each chunk down to the sentences that matter
    for the question. One LLM call per chunk.

    Adds compressed_text, which the generator then sends instead of the full
    text. A chunk with nothing relevant is left uncompressed rather than dropped;
    dropping is Corrective RAG's job.
    """
    compressed = []
    for chunk in chunks:
        prompt = f"""Copy, word for word, only the sentences from the text that help answer the question.
Do not add or rephrase anything. If no sentence helps, return exactly NONE.

Question: {question}

Text:
{chunk_text_for_llm(chunk)}"""
        extract = llm.complete(prompt, label="Contextual compression").strip()
        if extract and extract != "NONE":
            chunk = {**chunk, "compressed_text": extract}
        compressed.append(chunk)
    return compressed


def reposition(chunks: list[dict]) -> list[dict]:
    """Chunk positioning: LLMs attend most to the start and end of the context
    ("lost in the middle"), so put the best chunks there and the weakest in the
    middle. Expects chunks ordered best-first: 1 2 3 4 5 becomes 1 3 5 4 2.
    """
    return chunks[0::2] + chunks[1::2][::-1]
