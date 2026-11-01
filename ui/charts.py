"""Plotly charts for the app. Only visualisation lives here, no RAG logic."""
import textwrap

import numpy as np
import plotly.graph_objects as go

SOURCE_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756", "#72B7B2"]
TOP_COLOR = "#2CA02C"
OTHER_COLOR = "#B0B7C3"


def _wrap(text: str, width: int = 60) -> str:
    return "<br>".join(textwrap.wrap(text, width))


def _source_color_map(chunks: list[dict]) -> dict:
    sources = sorted({c["source"] for c in chunks})
    return {s: SOURCE_COLORS[i % len(SOURCE_COLORS)] for i, s in enumerate(sources)}


def vector_fingerprint(vector: list[float], title: str = "", columns: int = 96) -> go.Figure:
    """Draw a whole embedding as a grid of coloured cells (red = negative, blue = positive)."""
    v = np.array(vector)
    rows = int(np.ceil(len(v) / columns))
    grid = np.full(rows * columns, np.nan)
    grid[: len(v)] = v
    dims = np.arange(rows * columns).reshape(rows, columns)
    limit = float(np.max(np.abs(v))) or 1.0

    fig = go.Figure(go.Heatmap(
        z=grid.reshape(rows, columns), customdata=dims,
        colorscale="RdBu", zmid=0, zmin=-limit, zmax=limit, showscale=False,
        hovertemplate="dimension %{customdata}: %{z:.5f}<extra></extra>",
    ))
    fig.update_layout(
        title=title, height=max(160, rows * 9 + 60), margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(visible=False), yaxis=dict(visible=False, autorange="reversed"),
    )
    return fig


def all_embeddings_heatmap(chunks: list[dict], dims: int = 100) -> go.Figure:
    """Every row = one chunk, every column = one dimension of its embedding."""
    z = np.array([c["embedding"][:dims] for c in chunks])
    limit = float(np.max(np.abs(z))) or 1.0
    fig = go.Figure(go.Heatmap(
        z=z, x=list(range(dims)), y=[c["id"] for c in chunks],
        colorscale="RdBu", zmid=0, zmin=-limit, zmax=limit,
        hovertemplate="%{y}<br>dimension %{x}: %{z:.5f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"First {dims} dimensions of every stored embedding",
        xaxis_title="dimension", height=28 * len(chunks) + 120,
        margin=dict(l=10, r=10, t=50, b=40), yaxis=dict(autorange="reversed"),
    )
    return fig


def similarity_bars(scored: list[dict], top_ids: set = frozenset(), x_range: tuple | None = None) -> go.Figure:
    """One bar per chunk: its cosine similarity to the question."""
    colors = [TOP_COLOR if c["id"] in top_ids else OTHER_COLOR for c in scored]
    fig = go.Figure(go.Bar(
        x=[c["score"] for c in scored], y=[c["id"] for c in scored], orientation="h",
        marker_color=colors, text=[f"{c['score']:.4f}" for c in scored], textposition="outside",
        hovertext=[_wrap(c["text"]) for c in scored], hovertemplate="%{hovertext}<extra></extra>",
    ))
    fig.update_layout(
        title="Cosine similarity: question vs. each chunk",
        xaxis=dict(title="cosine similarity", range=x_range),
        yaxis=dict(autorange="reversed"), height=28 * max(len(scored), 4) + 120,
        margin=dict(l=10, r=10, t=50, b=40),
    )
    return fig


def embedding_map(chunks: list[dict], question_vector: list[float] | None = None, top_ids: list[str] = ()) -> go.Figure:
    """Squash every embedding to 2D with PCA so we can 'see' the meaning space."""
    matrix = np.array([c["embedding"] for c in chunks])
    mean = matrix.mean(axis=0)
    _, _, vt = np.linalg.svd(matrix - mean, full_matrices=False)
    project = lambda m: (np.atleast_2d(m) - mean) @ vt[:2].T
    points = project(matrix)

    fig = go.Figure()
    color_map = _source_color_map(chunks)
    for source, color in color_map.items():
        idx = [i for i, c in enumerate(chunks) if c["source"] == source]
        fig.add_trace(go.Scatter(
            x=points[idx, 0], y=points[idx, 1], mode="markers+text", name=source,
            text=[chunks[i]["id"] for i in idx], textposition="top center",
            marker=dict(size=14, color=color, line=dict(width=1, color="white")),
            hovertext=[_wrap(chunks[i]["text"]) for i in idx], hovertemplate="%{hovertext}<extra></extra>",
        ))

    if question_vector is not None:
        q = project(np.array(question_vector))[0]
        for chunk_id in top_ids:
            i = next(i for i, c in enumerate(chunks) if c["id"] == chunk_id)
            fig.add_trace(go.Scatter(
                x=[q[0], points[i, 0]], y=[q[1], points[i, 1]], mode="lines", showlegend=False,
                line=dict(color=TOP_COLOR, width=3, dash="dash"), hoverinfo="skip",
            ))
        fig.add_trace(go.Scatter(
            x=[q[0]], y=[q[1]], mode="markers+text", name="❓ your question", text=["QUESTION"],
            textposition="bottom center", marker=dict(size=26, color="#E45756", symbol="star"),
        ))

    fig.update_layout(
        title="Meaning map: every embedding squeezed into 2D (PCA)", height=520,
        xaxis=dict(showticklabels=False, title=""), yaxis=dict(showticklabels=False, title=""),
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig
