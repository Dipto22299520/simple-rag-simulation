"""STEP 8 - Generate: send the augmented prompt to the LLM."""
import requests

from rag.config import CHAT_MODEL, OPENROUTER_API_KEY, OPENROUTER_BASE_URL


def chat_request(messages: list[dict]) -> dict:
    """The exact JSON body we send to the LLM."""
    return {"model": CHAT_MODEL, "messages": messages, "temperature": 0}


def generate_answer(messages: list[dict], api_key: str = OPENROUTER_API_KEY) -> str:
    response = requests.post(
        f"{OPENROUTER_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=chat_request(messages),
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
