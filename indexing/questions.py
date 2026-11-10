"""Index questions instead of chunk text.

HyDE turns the question into a fake answer so it looks like the chunks. This
does the opposite at indexing time: an LLM writes the questions each chunk
answers, and those questions are what gets embedded. A user question is then
matched against other questions, and the hit points back to its chunk.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from generation.generator import LLM, parse_json_list


def _questions_for(chunk: dict, llm: LLM, n_questions: int) -> list[str]:
    prompt = f"""Write {n_questions} short questions that the text below answers.
Each question must make sense on its own, without seeing the text.
Return ONLY a JSON array of strings, nothing else.

Text:
{chunk["text"]}"""
    raw = llm.complete(prompt, max_tokens=300, label="Hypothetical questions")
    questions = [q.strip() for q in parse_json_list(raw) if isinstance(q, str) and q.strip()]
    return questions[:n_questions]


def generate_question_entries(chunks: list[dict], llm: LLM, n_questions: int, progress=None) -> list[dict]:
    """One index entry per generated question, each pointing back to its chunk
    through source_chunk_id / source_text. Costs one LLM call per chunk.

    A chunk the LLM returns no questions for is indexed by its own text, so it
    stays retrievable.
    """
    questions_per_chunk = [None] * len(chunks)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_questions_for, chunk, llm, n_questions): i for i, chunk in enumerate(chunks)}
        for done, future in enumerate(as_completed(futures), 1):
            questions_per_chunk[futures[future]] = future.result()
            if progress:
                progress(done / len(chunks))

    entries = []
    for chunk, questions in zip(chunks, questions_per_chunk):
        for j, text in enumerate(questions or [chunk["text"]]):
            entries.append({
                **chunk,
                "chunk_id": f"{chunk['chunk_id']}_q{j}",
                "text": text,
                "source_chunk_id": chunk["chunk_id"],
                "source_text": chunk["text"],
            })
    return entries
