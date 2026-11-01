"""STEP 3 - Turn text into an embedding (a list of numbers).

Texts with similar MEANING get vectors that point in similar directions.
We call the OpenRouter /embeddings endpoint with plain HTTP so you can
see exactly what is sent and what comes back.
"""
import hashlib
import re

import numpy as np
import requests

from rag.config import EMBEDDING_MODEL, OPENROUTER_API_KEY, OPENROUTER_BASE_URL

OFFLINE_MODEL_NAME = "offline-keyword-hash-256"


def embedding_request(text: str) -> dict:
    """The exact JSON body we send to the API."""
    return {"model": EMBEDDING_MODEL, "input": text}


def embed_text(text: str, api_key: str = OPENROUTER_API_KEY, offline: bool = False) -> list[float]:
    if offline:
        return offline_embedding(text)

    response = requests.post(
        f"{OPENROUTER_BASE_URL}/embeddings",
        headers={"Authorization": f"Bearer {api_key}"},
        json=embedding_request(text),
        timeout=30,
    )
    response.raise_for_status()
    # Response looks like: {"data": [{"embedding": [0.012, -0.034, ...]}], ...}
    return response.json()["data"][0]["embedding"]


def model_name(offline: bool) -> str:
    return OFFLINE_MODEL_NAME if offline else EMBEDDING_MODEL


def offline_embedding(text: str, dim: int = 256) -> list[float]:
    """Backup if there is no internet / API key.

    NOT a real semantic embedding: it only counts words (hashed into 256
    slots), so it matches keywords, not meaning. Good enough to demo.
    """
    vector = np.zeros(dim)
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        vector[h % dim] += 1 if (h >> 8) % 2 else -1
    norm = np.linalg.norm(vector)
    return (vector / norm).tolist() if norm else vector.tolist()
