"""All settings in one place. Change the models here."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- OpenRouter (one API key gives access to many models) ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Model that turns text into a vector (3072 numbers for this model)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "google/gemini-embedding-2")

# Model that writes the final answer
CHAT_MODEL = os.getenv("CHAT_MODEL", "google/gemini-3.1-flash-lite")

# --- Data ---
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "data" / "knowledge_base"
DB_PATH = PROJECT_ROOT / "data" / "vector_store.db"

# --- Retrieval ---
TOP_K = 3  # how many chunks we hand to the LLM
