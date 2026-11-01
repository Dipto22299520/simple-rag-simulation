"""STEP 6 - Retrieval: score every chunk against the question, keep the best K."""
from rag.similarity import cosine_similarity


def rank_chunks(question_vector: list[float], chunks: list[dict]) -> list[dict]:
    """Return all chunks with a 'score', best match first."""
    scored = [
        {**chunk, "score": cosine_similarity(question_vector, chunk["embedding"])}
        for chunk in chunks
    ]
    return sorted(scored, key=lambda c: c["score"], reverse=True)


def top_k(ranked: list[dict], k: int) -> list[dict]:
    return ranked[:k]
