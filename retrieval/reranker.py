from functools import lru_cache

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def _load_cross_encoder(model_name: str):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_name)


def rerank(question: str, chunks: list[dict], top_n: int, model_name: str = RERANK_MODEL) -> list[dict]:
    """Re-score chunks with a cross-encoder, which reads question and chunk
    together and is more accurate (and much slower) than comparing embeddings.

    Returns the top_n chunks with score replaced by the cross-encoder's 0-1
    relevance; the score retrieval gave is kept as retrieval_score.
    """
    if not chunks:
        return []
    import torch
    scores = _load_cross_encoder(model_name).predict(
        [(question, chunk["text"]) for chunk in chunks],
        activation_fn=torch.nn.Sigmoid(),  # the model outputs logits
        show_progress_bar=False,
    )
    reranked = [
        {
            **chunk,
            "score": round(float(score), 4),
            "score_type": "rerank",
            "retrieval_score": chunk["score"],
            "retrieval_score_type": chunk["score_type"],
        }
        for chunk, score in zip(chunks, scores)
    ]
    reranked.sort(key=lambda c: c["score"], reverse=True)
    return reranked[:top_n]
