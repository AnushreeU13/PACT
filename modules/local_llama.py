"""
Local Llama wrapper for PACT - Groq backend (cloud deployment, no Ollama required).

Uses the Groq API for text generation (llama-3.1-8b-instant).
AU-Probe uncertainty is computed using a linear probe trained on Llama embeddings
and distilled into a sentence-transformer (all-MiniLM-L6-v2) backbone for deployment.
The probe applies: score = sigmoid(w . embedding + b) where w and b were learned
by distilling the original Llama-3.1-8b layer-32 probe onto MiniLM embeddings.

Required environment variable:
    GROQ_API_KEY  - obtain a free key at https://console.groq.com
"""

from __future__ import annotations

import math
import os
from typing import Optional

import torch
from groq import Groq

DEFAULT_MODEL_NAME = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

_groq_client: Optional[Groq] = None
_loaded_model_name: Optional[str] = None
_groq_ready: bool = False

# AU-Probe state
_probe_w: Optional[torch.Tensor] = None   # shape [384]
_probe_b: float = 0.0
_probe_ready: bool = False
_st_model = None                           # SentenceTransformer, lazy-loaded


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
        "probe_ready":       _probe_ready,
        "backend":           "groq",
        "groq_api_key_set":  api_key_set,
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

    _groq_client = None
    _get_client()

    _loaded_model_name = model_name
    _groq_ready = True
    print(f"Groq backend ready: model={model_name}")


# ---------------------------------------------------------------------------
# AU-Probe - MiniLM linear probe (distilled from original Llama layer-32 probe)
# ---------------------------------------------------------------------------

def _get_st_model():
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("AU probe: MiniLM sentence-transformer loaded.")
    return _st_model


def load_au_probe(probe_path: str, layer: int = 0) -> None:
    """
    Load the MiniLM-distilled linear probe from a .pt file.
    Falls back to the text heuristic if the file is not found.
    """
    global _probe_w, _probe_b, _probe_ready

    if not probe_path or not os.path.isfile(probe_path):
        # Try locating minilm_probe.pt relative to this file
        candidate = os.path.join(
            os.path.dirname(__file__), "..", "data", "au_probe", "minilm_probe.pt"
        )
        candidate = os.path.abspath(candidate)
        if os.path.isfile(candidate):
            probe_path = candidate
        else:
            print("AU probe: minilm_probe.pt not found - falling back to text heuristic.")
            return

    try:
        data = torch.load(probe_path, map_location="cpu", weights_only=False)
        _probe_w = data["w"].float().squeeze()   # [384]
        _probe_b = float(data["b"])
        _probe_ready = True
        print(f"AU probe: MiniLM probe loaded (w={_probe_w.shape}, b={_probe_b:.4f}).")
        # Pre-load the sentence-transformer so first request is fast
        _get_st_model()
    except Exception as e:
        print(f"AU probe: failed to load probe file ({e}) - falling back to text heuristic.")
        _probe_ready = False


def _text_heuristic(prompt: str) -> float:
    """Fallback: redaction-ratio sigmoid when the probe file is unavailable."""
    import re
    if not prompt.strip():
        return 0.0
    redacted = len(re.findall(r'\[REDACTED[^\]]*\]', prompt, re.IGNORECASE))
    total = len(prompt.split())
    if total == 0:
        return 0.0
    ratio = redacted / total
    return round(1.0 / (1.0 + math.exp(-10.0 * (ratio - 0.30))), 4)


def get_au_uncertainty(
    prompt: str,
    model_name: str = DEFAULT_MODEL_NAME,
    use_chat_template: bool = False,
) -> float:
    """
    Estimate uncertainty of the redacted prompt using the MiniLM linear probe.

    Gets a 384-d sentence-transformer embedding, then applies:
        score = sigmoid(w . embedding + b)
    where w and b were distilled from the original Llama-3.1-8b layer-32 probe.

    Falls back to the text heuristic if the probe is not loaded.
    Returns a float in [0, 1].
    """
    if not prompt.strip():
        return 0.0

    if not _probe_ready or _probe_w is None:
        score = _text_heuristic(prompt)
        print(f"DEBUG: AU heuristic (fallback) score={score:.4f}")
        return score

    try:
        st = _get_st_model()
        embedding = st.encode(prompt, normalize_embeddings=True, show_progress_bar=False)
        emb_tensor = torch.tensor(embedding, dtype=torch.float32)

        w = _probe_w.cpu()
        if emb_tensor.shape[0] != w.shape[0]:
            # Dimension mismatch safeguard
            n = min(emb_tensor.shape[0], w.shape[0])
            emb_tensor = emb_tensor[:n]
            w = w[:n]

        logit = torch.dot(w, emb_tensor).item() + _probe_b
        score = round(1.0 / (1.0 + math.exp(-logit)), 4)
        print(f"DEBUG: AU MiniLM probe score={score:.4f}")
        return score

    except Exception as e:
        print(f"DEBUG: AU probe error ({e}), falling back to heuristic.")
        return _text_heuristic(prompt)


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
