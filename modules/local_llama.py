"""
Local Llama wrapper for PACT - Groq backend (cloud deployment, no Ollama required).

Uses the Groq API for text generation (llama-3.1-8b-instant).
AU-Probe uncertainty is computed via a text heuristic instead of a neural probe:
counts [REDACTED ...] tokens in the final prompt as a fraction of total words,
then applies a sigmoid to produce a 0-1 uncertainty score.

Required environment variable:
    GROQ_API_KEY  - obtain a free key at https://console.groq.com
"""

from __future__ import annotations

import math
import os
import re
from typing import Optional

from groq import Groq

DEFAULT_MODEL_NAME = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

_groq_client: Optional[Groq] = None
_loaded_model_name: Optional[str] = None
_groq_ready: bool = False


# ---------------------------------------------------------------------------
# Groq client helpers
# ---------------------------------------------------------------------------

def _get_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY environment variable is not set. "
                "Get a free key at https://console.groq.com and set it before starting the server."
            )
        _groq_client = Groq(api_key=api_key)
    return _groq_client


# ---------------------------------------------------------------------------
# Public model-lifecycle API (same interface as the Ollama version)
# ---------------------------------------------------------------------------

def is_loaded() -> bool:
    return _groq_ready and _loaded_model_name is not None


def get_status() -> dict:
    api_key_set = bool(os.environ.get("GROQ_API_KEY", "").strip())
    return {
        "loaded":            is_loaded(),
        "model_name":        _loaded_model_name,
        "probe_ready":       False,
        "backend":           "groq",
        "groq_api_key_set":  api_key_set,
        # kept for UI compatibility
        "cuda_available":    False,
        "cuda_device_count": 0,
        "cuda_device_name":  None,
    }


def load_model(
    model_name: str = DEFAULT_MODEL_NAME,
    force_reload: bool = False,
) -> None:
    """
    Verify the Groq API key is present and the client can be instantiated.
    No weights are downloaded - Groq runs inference on their servers.
    """
    global _loaded_model_name, _groq_ready, _groq_client

    if not force_reload and _groq_ready and _loaded_model_name == model_name:
        return

    _groq_client = None  # reset so _get_client() re-reads the env var
    _get_client()        # raises if GROQ_API_KEY is missing

    _loaded_model_name = model_name
    _groq_ready = True
    print(f"Groq backend ready: model={model_name}")


# ---------------------------------------------------------------------------
# AU-Probe - heuristic implementation
# ---------------------------------------------------------------------------
# The original neural probe needs Llama hidden states (4096-d vectors).
# Groq does not expose hidden states, so we use a text-based proxy instead:
#   ratio = count([REDACTED...] tokens) / total words in final prompt
#   score = sigmoid(10 * (ratio - 0.30))
# At ratio=0.30 → score≈0.5; at ratio=0.45 → score≈0.82 (above threshold).

_REDACTED_PATTERN = re.compile(r'\[REDACTED[^\]]*\]', re.IGNORECASE)


def load_au_probe(probe_path: str, layer: int = 32) -> None:
    """No-op in the Groq deployment version - heuristic probe needs no file."""
    print("AU probe: using text heuristic (Groq deployment mode, no probe file needed).")


def get_au_uncertainty(
    prompt: str,
    model_name: str = DEFAULT_MODEL_NAME,
    use_chat_template: bool = False,
) -> float:
    """
    Estimate uncertainty of the redacted prompt via a token-ratio heuristic.

    Counts [REDACTED ...] markers as a fraction of total words, then maps
    through a sigmoid centred at 30% redaction. Returns a float in [0, 1].
    Threshold check (>= 0.8) lives in server.py.
    """
    if not prompt.strip():
        return 0.0

    redacted_count = len(_REDACTED_PATTERN.findall(prompt))
    total_words = len(prompt.split())

    if total_words == 0:
        return 0.0

    ratio = redacted_count / total_words
    score = 1.0 / (1.0 + math.exp(-10.0 * (ratio - 0.30)))
    score = round(score, 4)
    print(f"DEBUG: AU heuristic score={score:.4f}  redacted={redacted_count}/{total_words} words ({ratio:.0%})")
    return score


# ---------------------------------------------------------------------------
# Text generation via Groq
# ---------------------------------------------------------------------------

def generate_text(
    prompt: str,
    max_new_tokens: int = 220,
    temperature: float = 0.2,
    top_p: float = 0.9,
    model_name: str = DEFAULT_MODEL_NAME,
    use_chat_template: bool = False,
    repetition_penalty: float = 1.12,
) -> str:
    """
    Generate text via the Groq API and return the model's reply.
    `use_chat_template=True` wraps the prompt as a user chat message (recommended).
    """
    client = _get_client()
    active_model = model_name if model_name != DEFAULT_MODEL_NAME else (_loaded_model_name or DEFAULT_MODEL_NAME)

    if use_chat_template:
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

    completion = client.chat.completions.create(
        model=active_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_new_tokens,
        top_p=top_p,
    )
    return (completion.choices[0].message.content or "").strip()
