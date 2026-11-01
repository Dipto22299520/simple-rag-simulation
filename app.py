"""RAG, step by step - a live visual walkthrough.

Run with:   streamlit run app.py

This file is ONLY the user interface. Every RAG step lives in rag/:
    loader -> chunker -> embedder -> vector_db -> similarity/retriever -> prompt_builder -> generator
"""
import json
import time

import pandas as pd
import streamlit as st

from rag import config, vector_db
from rag.chunker import chunk_documents
from rag.embedder import embed_text, embedding_request, model_name
from rag.generator import chat_request, generate_answer
from rag.loader import load_documents
from rag.prompt_builder import build_messages, build_messages_without_rag
from rag.retriever import rank_chunks, top_k
from rag.similarity import cosine_breakdown
from ui import charts

EXAMPLE_QUESTIONS = [
    "When does the library close on Saturday?",
    "Can I use my phone as a calculator in the exam?",
    "What GPA do I need for the Aurora scholarship?",
    "How much does it cost to park my car on campus?",
    "What's the Wi-Fi password?",
    "What is the capital of France?",
]

st.set_page_config(page_title="RAG Step by Step", page_icon="🔎", layout="wide")

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("OpenRouter API key", value=config.OPENROUTER_API_KEY, type="password")
    offline = st.toggle(
        "Offline embeddings (no API)", value=not api_key,
        help="Backup mode: a simple keyword-hash 'embedding' computed locally. Matches words, not meaning.",
    )
    k = st.slider("Top-K chunks to retrieve", 1, 6, config.TOP_K)
    delay = st.slider("Animation delay (seconds)", 0.0, 1.0, 0.3, 0.05)
    st.divider()
    st.markdown(f"**Embedding model**  \n`{model_name(offline)}`")
    st.markdown(f"**Chat model**  \n`{config.CHAT_MODEL}`")


def pause(seconds: float | None = None) -> None:
    time.sleep(delay if seconds is None else seconds)


def short_vector(vector: list[float], n: int = 8) -> str:
    return "[" + ", ".join(f"{x:+.5f}" for x in vector[:n]) + f", ... ]   <- {len(vector)} numbers in total"


def show_api_error(error: Exception) -> None:
    st.error(f"API call failed: {error}\n\nCheck the API key, or switch on **Offline embeddings** in the sidebar.")


# ---------------------------------------------------------------- header
st.title("🔎 Retrieval-Augmented Generation, step by step")
st.markdown(
    "**Part 1 · Indexing** (done once): 📄 Documents → ✂️ Chunks → 🔢 Embeddings → 🗄️ Vector database  \n"
    "**Part 2 · Querying** (every question): ❓ Question → 🔢 Embedding → 📐 Cosine similarity → "
    "🏆 Top-K chunks → 📝 Prompt → 🤖 LLM answer"
)
tab_index, tab_query = st.tabs(["📚 Part 1 · Build the knowledge base", "❓ Part 2 · Ask a question"])


# ================================================================ PART 1
with tab_index:
    documents = load_documents()
    chunks = chunk_documents(documents)
    current_model = model_name(offline)

    # ---- Step 1
    st.header("Step 1 · The documents")
    st.caption(f"{len(documents)} text files in `data/knowledge_base/`. Greenfield College is made up, "
               "so the LLM has never seen these facts.")
    for col, doc in zip(st.columns(len(documents)), documents):
        with col, st.container(border=True, height=300):
            st.markdown(f"**📄 {doc['source']}**")
            st.write(doc["text"])

    # ---- Step 2
    st.header("Step 2 · Chunking")
    st.caption("Rule: every paragraph becomes one chunk. Small chunks = precise matches.")
    st.dataframe(
        pd.DataFrame([{"chunk id": c["id"], "source": c["source"], "chars": len(c["text"]), "text": c["text"]}
                      for c in chunks]),
        hide_index=True,
    )

    # ---- Step 3
    st.header("Step 3 · Embed every chunk and store it in the database")
    with st.expander("🔌 What we send to the embedding API for each chunk"):
        st.code(f"POST {config.OPENROUTER_BASE_URL}/embeddings", language="text")
        st.json(embedding_request(chunks[0]["text"]))

    stored = vector_db.fetch_all()
    if stored and stored[0]["model"] == current_model and len(stored) == len(chunks):
        st.success(f"The database already holds {len(stored)} chunks embedded with `{current_model}`. "
                   "Click the button to rebuild it live.")

    if st.button("▶️ Embed all chunks and store them in the database", type="primary"):
        vector_db.clear()
        progress = st.progress(0.0)
        live = st.empty()
        for i, chunk in enumerate(chunks):
            try:
                vector = embed_text(chunk["text"], api_key, offline)
            except Exception as error:
                show_api_error(error)
                break
            vector_db.insert_chunk(chunk, vector, current_model)

            with live.container(border=True):
                left, right = st.columns(2)
                with left:
                    st.markdown(f"**Chunk `{chunk['id']}`**")
                    st.info(chunk["text"])
                    st.markdown("⬇️ *embedding model* ⬇️")
                    st.code(short_vector(vector), language="text", wrap_lines=True)
                    st.caption("✅ saved to `data/vector_store.db`")
                with right:
                    st.plotly_chart(charts.vector_fingerprint(vector, "The embedding as a picture (1 cell = 1 number)"),
                                    key=f"live-fp-{i}")
            progress.progress((i + 1) / len(chunks), text=f"Embedded {i + 1} / {len(chunks)} chunks")
            pause()
        else:
            st.success("Done! Every chunk is now a vector in the database.")
        stored = vector_db.fetch_all()

    # ---- Step 4
    st.header("Step 4 · Look inside the vector database")
    if not stored:
        st.warning("The database is empty. Click the button in Step 3.")
    else:
        st.code("SELECT id, source, text, embedding, model FROM chunks;", language="sql")
        st.dataframe(
            pd.DataFrame([{"id": r[0], "source": r[1], "text": r[2],
                           "embedding (as stored in the DB)": r[3][:100] + " ...", "model": r[4]}
                          for r in vector_db.fetch_raw()]),
            hide_index=True,
        )

        picked_id = st.selectbox("Pick a chunk to see its FULL embedding", [c["id"] for c in stored])
        picked = next(c for c in stored if c["id"] == picked_id)
        left, right = st.columns(2)
        with left:
            st.info(picked["text"])
            st.caption(f"{len(picked['embedding'])} numbers. Scroll through them all:")
            st.code(json.dumps([round(x, 6) for x in picked["embedding"]]), language="json",
                    wrap_lines=True, height=260)
        with right:
            st.plotly_chart(charts.vector_fingerprint(picked["embedding"], f"Fingerprint of {picked_id}"))

        st.plotly_chart(charts.all_embeddings_heatmap(stored))
        st.caption("Every row is a chunk. The individual numbers mean nothing to a human. "
                   "Only the overall *direction* of the vector carries meaning.")

        st.plotly_chart(charts.embedding_map(stored))
        st.caption("PCA squeezes each vector down to just 2 numbers. Chunks about similar topics end up close together.")


# ================================================================ PART 2
def run_query_pipeline(result: dict, stored: list[dict], animate: bool) -> None:
    """Show steps 5-9. Values are computed once and cached in `result`;
    if `animate` is True the steps are revealed one by one."""
    wait = pause if animate else (lambda *_: None)
    question, k_used = result["question"], result["k"]

    # ---- Step 5
    st.header("Step 5 · Embed the question (same model as the chunks)")
    st.markdown(f"> ❓ **{question}**")
    if "q_vec" not in result:
        try:
            with st.spinner("Calling the embedding model..."):
                result["q_vec"] = embed_text(question, api_key, offline)
        except Exception as error:
            show_api_error(error)
            return
    q_vec = result["q_vec"]
    left, right = st.columns(2)
    with left:
        st.code(short_vector(q_vec), language="text", wrap_lines=True)
        st.caption("Now the question lives in the same vector space as every chunk, so we can measure distances.")
    with right:
        st.plotly_chart(charts.vector_fingerprint(q_vec, "Question fingerprint"))
    wait()

    # ---- Step 6
    st.header("Step 6 · Compare the question with EVERY chunk (cosine similarity)")
    st.latex(r"\text{cosine}(\vec q,\vec d)=\frac{\vec q\cdot\vec d}{\lVert\vec q\rVert\;\lVert\vec d\rVert}"
             r"=\frac{\sum_i q_i d_i}{\sqrt{\sum_i q_i^2}\;\sqrt{\sum_i d_i^2}}")
    ranked = rank_chunks(q_vec, stored)
    top_ids = {c["id"] for c in ranked[:k_used]}
    scores = [c["score"] for c in ranked]
    x_range = (min(0.0, min(scores)) - 0.05, max(scores) + 0.12)

    chart_slot = st.empty()
    if animate:  # reveal one score at a time, in database order
        score_of = {c["id"]: c["score"] for c in ranked}
        for n in range(1, len(stored) + 1):
            revealed = [{**c, "score": score_of[c["id"]]} for c in stored[:n]]
            chart_slot.plotly_chart(charts.similarity_bars(revealed, x_range=x_range), key=f"sim-{n}")
            pause(delay / 3)
        pause()
    chart_slot.plotly_chart(charts.similarity_bars(ranked, top_ids, x_range), key="sim-final")
    st.caption(f"Sorted from most to least similar. Green = the top {k_used} that will be retrieved.")

    with st.expander("🔬 Show the math for the #1 chunk", expanded=True):
        b = cosine_breakdown(q_vec, ranked[0]["embedding"])
        st.markdown(f"Question vs. **`{ranked[0]['id']}`**. The first {len(b['first_dims'])} of {b['dims']} dimensions:")
        st.dataframe(
            pd.DataFrame(b["first_dims"]).rename(columns={"question": "q_i", "chunk": "d_i", "product": "q_i × d_i"}),
            hide_index=True,
        )
        st.markdown(
            f"- **dot product** (add up all {b['dims']} products) = `{b['dot_product']:.5f}`\n"
            f"- **|q|** = `{b['norm_question']:.5f}`  ·  **|d|** = `{b['norm_chunk']:.5f}`\n"
            f"- **cosine** = {b['dot_product']:.5f} / ({b['norm_question']:.5f} × {b['norm_chunk']:.5f}) "
            f"= **`{b['cosine']:.5f}`**"
        )
        if abs(b["norm_question"] - 1) < 0.01 and abs(b["norm_chunk"] - 1) < 0.01:
            st.caption("These vectors already have length 1, so here the cosine is simply the dot product.")
    wait()

    # ---- Step 7
    st.header(f"Step 7 · Retrieve the top {k_used} chunks")
    retrieved = top_k(ranked, k_used)
    for rank, chunk in enumerate(retrieved, start=1):
        st.success(f"**#{rank} · `{chunk['id']}` · similarity {chunk['score']:.4f}**\n\n{chunk['text']}")
        wait()
    st.plotly_chart(charts.embedding_map(stored, q_vec, [c["id"] for c in retrieved]))
    st.caption("The red star is the question. Dashed lines point to the retrieved chunks. "
               "(2D is only an approximation; the bar chart above shows the real scores.)")
    wait()

    # ---- Step 8
    st.header("Step 8 · Build the prompt. This is EXACTLY what goes into the LLM")
    messages = build_messages(question, retrieved)
    left, right = st.columns(2)
    with left:
        for message in messages:
            st.markdown(f"**{message['role'].upper()} message**")
            st.code(message["content"], language="text", wrap_lines=True)
    with right:
        st.markdown(f"**Raw JSON body** sent to `POST {config.OPENROUTER_BASE_URL}/chat/completions`")
        st.json(chat_request(messages))
    wait()

    # ---- Step 9
    st.header("Step 9 · The LLM writes the answer")
    if "answer" not in result:
        try:
            with st.spinner(f"Asking {config.CHAT_MODEL}..."):
                result["answer"] = generate_answer(messages, api_key)
        except Exception as error:
            show_api_error(error)
            return
    st.chat_message("assistant").markdown(result["answer"])

    st.divider()
    if st.button("🙈 Compare: ask the same question WITHOUT retrieval"):
        try:
            with st.spinner("Asking the LLM with no context..."):
                result["no_rag_answer"] = generate_answer(build_messages_without_rag(question), api_key)
        except Exception as error:
            show_api_error(error)
    if "no_rag_answer" in result:
        with_rag, without_rag = st.columns(2)
        with with_rag:
            st.markdown("**✅ With RAG**")
            st.chat_message("assistant").markdown(result["answer"])
        with without_rag:
            st.markdown("**❌ Without RAG** (the model only knows its training data)")
            st.chat_message("assistant").markdown(result["no_rag_answer"])


with tab_query:
    stored = vector_db.fetch_all()
    if not stored:
        st.warning("The database is empty. Build it in Part 1 first.")
    elif stored[0]["model"] != model_name(offline):
        st.warning(f"The database was built with `{stored[0]['model']}`, but the current model is "
                   f"`{model_name(offline)}`. Vectors from different models can't be compared, so rebuild in Part 1.")
    else:
        example = st.selectbox("Pick an example question...", EXAMPLE_QUESTIONS)
        question = st.text_input("...or type your own", value=example)
        run = st.button("🚀 Run retrieval", type="primary")
        if run:
            st.session_state.result = {"question": question, "k": k}
        if "result" in st.session_state:
            run_query_pipeline(st.session_state.result, stored, animate=run)
