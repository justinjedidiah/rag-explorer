import json
import re

def handle_query(question: str, strategy: str, llm_fn) -> dict:
    """
    Returns:
      {
        "queries":       list[str],  # one or more queries to search with
        "display":       dict,       # debug info for the UI
        "strategy":      str,
        "original":      str,
      }
    """
    if strategy == "None (raw question)":
        return {
            "queries": [question],
            "display": {},
            "strategy": strategy,
            "original": question,
        }

    elif strategy == "Rewrite":
        prompt = f"""Rewrite this question to be clearer and more specific for searching a document.
Return ONLY the rewritten question, nothing else.

Original: {question}
Rewritten:"""
        rewritten = llm_fn(prompt).strip()
        return {
            "queries": [rewritten],
            "display": {"rewritten": rewritten},
            "strategy": strategy,
            "original": question,
        }

    elif strategy == "HyDE":
        prompt = f"""Write a short factual paragraph (3-5 sentences) that would directly answer this question.
Write as if you found this text in a document. No preamble, just the paragraph.

Question: {question}
Answer paragraph:"""
        hypothetical = llm_fn(prompt).strip()
        return {
            "queries": [hypothetical],
            "display": {"hypothetical_document": hypothetical},
            "strategy": strategy,
            "original": question,
        }

    elif strategy == "Decompose":
        prompt = f"""Break this question into 2-4 simpler sub-questions.
Return ONLY a JSON array of strings, nothing else.
Example: ["What is X?", "How does Y work?"]

Question: {question}
Sub-questions:"""
        raw = llm_fn(prompt).strip()
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        try:
            sub_questions = json.loads(match.group()) if match else [question]
        except json.JSONDecodeError:
            sub_questions = [question]
        return {
            "queries": sub_questions,
            "display": {"sub_questions": sub_questions},
            "strategy": strategy,
            "original": question,
        }