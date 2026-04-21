from __future__ import annotations

import os


DEFAULT_OLLAMA_MODEL = "llama3.2:1b"


def get_ollama_model() -> str:
    return os.getenv("PHASE2_OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL).strip() or DEFAULT_OLLAMA_MODEL
