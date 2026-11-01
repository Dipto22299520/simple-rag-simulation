"""STEP 5 - Cosine similarity: how close are two vectors in direction?

    cos(q, d) = (q . d) / (|q| * |d|)

     1.0  -> same direction  (very similar meaning)
     0.0  -> unrelated
    -1.0  -> opposite
"""
import numpy as np


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    dot_product = np.dot(a, b)                       # sum of a[i] * b[i]
    lengths = np.linalg.norm(a) * np.linalg.norm(b)  # |a| * |b|
    return float(dot_product / lengths)


def cosine_breakdown(a: list[float], b: list[float], show_dims: int = 6) -> dict:
    """Same math as above, but every intermediate number is kept for teaching."""
    a, b = np.array(a), np.array(b)
    products = a * b
    return {
        "first_dims": [
            {"dim": i, "question": a[i], "chunk": b[i], "product": products[i]}
            for i in range(show_dims)
        ],
        "dims": len(a),
        "dot_product": float(products.sum()),
        "norm_question": float(np.linalg.norm(a)),
        "norm_chunk": float(np.linalg.norm(b)),
        "cosine": cosine_similarity(a, b),
    }
