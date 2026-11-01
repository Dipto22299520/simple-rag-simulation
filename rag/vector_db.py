"""STEP 4 - Store the vectors in a database.

Real projects use a vector database (Chroma, Pinecone, pgvector, ...).
Here we use plain SQLite so you can open the file and SEE the rows:
each row holds the chunk text AND its embedding (stored as JSON text).
"""
import json
import sqlite3

from rag.config import DB_PATH

TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id        TEXT PRIMARY KEY,
    source    TEXT,
    text      TEXT,
    embedding TEXT,   -- the vector, e.g. "[0.0123, -0.0456, ...]"
    model     TEXT    -- which embedding model produced it
)
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(TABLE_SCHEMA)
    return conn


def clear() -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM chunks")


def insert_chunk(chunk: dict, embedding: list[float], model: str) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO chunks (id, source, text, embedding, model) VALUES (?, ?, ?, ?, ?)",
            (chunk["id"], chunk["source"], chunk["text"], json.dumps(embedding), model),
        )


def fetch_all() -> list[dict]:
    """Load every chunk with its embedding turned back into a list of floats."""
    with _connect() as conn:
        rows = conn.execute("SELECT id, source, text, embedding, model FROM chunks ORDER BY id").fetchall()
    return [
        {"id": r[0], "source": r[1], "text": r[2], "embedding": json.loads(r[3]), "model": r[4]}
        for r in rows
    ]


def fetch_raw(limit: int = 100) -> list[tuple]:
    """Rows exactly as they sit in the database (embedding is still a string)."""
    with _connect() as conn:
        return conn.execute("SELECT id, source, text, embedding, model FROM chunks ORDER BY id LIMIT ?", (limit,)).fetchall()


def count() -> int:
    with _connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
