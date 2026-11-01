"""STEP 1 - Load the documents that make up our knowledge base."""
from pathlib import Path

from rag.config import KNOWLEDGE_BASE_DIR


def load_documents(folder: Path = KNOWLEDGE_BASE_DIR) -> list[dict]:
    """Read every .txt file in the folder. Each document = {source, text}."""
    documents = []
    for path in sorted(folder.glob("*.txt")):
        documents.append({
            "source": path.name,
            "text": path.read_text(encoding="utf-8").strip(),
        })
    return documents
