"""STEP 7 - Augment: paste the retrieved chunks into the prompt.

This is the "A" in RAG. The LLM never searches anything itself.
It just receives a prompt that already contains the right context.
"""

SYSTEM_PROMPT = (
    "You are a helpful assistant for Greenfield College students. "
    "Answer the question using ONLY the context below. "
    "If the answer is not in the context, say \"I don't know based on the provided documents.\""
)


def build_messages(question: str, retrieved_chunks: list[dict]) -> list[dict]:
    context = "\n\n".join(
        f"[{i}] (from {c['source']})\n{c['text']}" for i, c in enumerate(retrieved_chunks, start=1)
    )
    user_prompt = f"CONTEXT:\n{context}\n\nQUESTION: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_messages_without_rag(question: str) -> list[dict]:
    """For comparison: the same question with no retrieved context."""
    return [{"role": "user", "content": question}]
