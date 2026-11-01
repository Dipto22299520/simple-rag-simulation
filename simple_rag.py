"""The WHOLE RAG pipeline in one short script, no UI. Run:  python simple_rag.py

Uses the same rag/ modules as the visual app. Good for showing students
that RAG is really just these few lines.
"""
from rag import vector_db
from rag.chunker import chunk_documents
from rag.embedder import EMBEDDING_MODEL, embed_text
from rag.generator import generate_answer
from rag.loader import load_documents
from rag.prompt_builder import build_messages
from rag.retriever import rank_chunks, top_k

# ---------- Part 1: indexing (done once) ----------
documents = load_documents()                                 # 1. load
chunks = chunk_documents(documents)                          # 2. chunk
vector_db.clear()
for chunk in chunks:
    vector = embed_text(chunk["text"])                       # 3. embed
    vector_db.insert_chunk(chunk, vector, EMBEDDING_MODEL)   # 4. store
print(f"Stored {vector_db.count()} chunks.\n")

# ---------- Part 2: querying (every question) ----------
question = input("Ask a question: ")
question_vector = embed_text(question)                       # 5. embed the question
ranked = rank_chunks(question_vector, vector_db.fetch_all()) # 6. cosine similarity vs. every chunk
retrieved = top_k(ranked, 3)                                 # 7. keep the best 3

print("\nRetrieved chunks:")
for chunk in retrieved:
    print(f"  {chunk['score']:.3f}  {chunk['text'][:80]}...")

messages = build_messages(question, retrieved)               # 8. augment the prompt
print("\nAnswer:", generate_answer(messages))                # 9. generate
