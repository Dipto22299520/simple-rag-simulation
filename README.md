# RAG, Step by Step: a live classroom demo

A visual simulation of Retrieval-Augmented Generation. Every step of the pipeline
is shown on screen: real embeddings, the rows in the database, cosine similarity
scores, the retrieved chunks and the exact prompt sent to the LLM.

## Run it

```bash
pip install -r requirements.txt
copy .env.example .env        # then paste your OpenRouter key into .env
streamlit run app.py
```

Get a free key at [openrouter.ai/keys](https://openrouter.ai/keys). One key covers both models used here:
`google/gemini-embedding-2` for embeddings and `google/gemini-3.1-flash-lite` for answers (change them in `rag/config.py`).
You can also paste the key into the app's sidebar instead of using `.env`.
No key or no internet? Switch on **Offline embeddings** in the sidebar. A simple keyword-based
vector is computed locally so Steps 1-8 still work, but Step 9 (the LLM answer) needs the key.

## Code map: one file per RAG step

| Step | File | What it does |
|---|---|---|
| 1 | `rag/loader.py` | Reads the `.txt` files in `data/knowledge_base/` |
| 2 | `rag/chunker.py` | Splits each document into paragraphs (chunks) |
| 3 | `rag/embedder.py` | Text → vector via OpenRouter `/embeddings` |
| 4 | `rag/vector_db.py` | Stores text + vector in SQLite (`data/vector_store.db`) |
| 5 | `rag/similarity.py` | Cosine similarity, written out by hand |
| 6 | `rag/retriever.py` | Scores every chunk, keeps the top K |
| 7 | `rag/prompt_builder.py` | Pastes the retrieved chunks into the prompt |
| 8 | `rag/generator.py` | Sends the prompt to the LLM via OpenRouter |
| - | `rag/config.py` | API key, model names, paths |
| - | `ui/charts.py` | Plotly charts (visualisation only) |
| - | `app.py` | The Streamlit UI that shows each step |
| - | `simple_rag.py` | The whole pipeline in about 30 lines, no UI (`python simple_rag.py`) |

## Suggested 15-minute flow

1. **Part 1 tab, Steps 1-2.** Show the documents (fictional college, so the LLM can't already know it), then the chunks.
2. **Step 3.** Click *Embed all chunks*. Each chunk turns into 3072 numbers live.
3. **Step 4.** Show the raw DB rows. The embedding is just a long list of numbers stored as text.
   Pick a chunk and scroll through its full vector. Show the meaning map: same-topic chunks cluster together.
4. **Part 2 tab.** Ask *"Can I use my phone as a calculator in the exam?"* and click **Run retrieval**.
   Walk through question embedding → similarity bars filling in → the cosine math → top-K → the exact prompt → the answer.
5. Click **Compare: without RAG**. The model alone can't know Greenfield's rules.
6. Ask *"What is the capital of France?"*. Similarity scores are low, and the model says it doesn't know
   *based on the documents*. That shows the LLM only sees what retrieval gives it.

Turn the **Animation delay** slider up for a slower, more dramatic reveal.
