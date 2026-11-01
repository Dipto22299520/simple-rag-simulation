"""STEP 2 - Split documents into small chunks.

Why? An embedding of a whole book is too "blurry" to match a specific
question. Small, focused pieces of text are easier to match.
Our strategy is the simplest one: one paragraph = one chunk.
"""


def chunk_documents(documents: list[dict]) -> list[dict]:
    chunks = []
    for doc in documents:
        paragraphs = [p.strip() for p in doc["text"].split("\n\n") if p.strip()]
        doc_name = doc["source"].removesuffix(".txt")
        for i, paragraph in enumerate(paragraphs):
            chunks.append({
                "id": f"{doc_name}-{i}",
                "source": doc["source"],
                "text": paragraph,
            })
    return chunks
