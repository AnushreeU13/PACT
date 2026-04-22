from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
import uvicorn
import sys
import os
import json
import requests
import threading
import time

# Import our financial detector
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from modules import local_llama
from modules.pipeline_collect import collect_pipeline_inputs
from modules.synthesis_prompt import (
    build_privacy_synthesis_prompt,
    extract_final_prompt,
    is_synthesis_unusable,
)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
QUERIES_JSON_PATH = os.path.join(ROOT_DIR, 'data', 'queries.json')

app = FastAPI()

def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            "Set it before starting the backend."
        )
    return value

# These keys are only used once we wire up real local LLM (HF) and cloud GPT calls.
# Failing fast here makes misconfiguration obvious.
#
# We no longer require HF_API_KEY because local Llama runs on the machine.
# (If your llama wrapper needs an HF token, it will use HF_TOKEN/HUGGING_FACE_HUB_TOKEN
# or hg.txt/hg files as implemented in llama_model.py.)
GPT_API_KEY = os.environ.get("GPT_API_KEY", "")

GPT_MODEL_ID = "gpt-4o-mini"

LOCAL_LLM_MODEL_NAME = os.environ.get(
    "LOCAL_LLM_MODEL_NAME",
    "llama-3.1-8b-instant",
)


def _env_truthy(name: str, default: str = "1") -> bool:
    v = (os.environ.get(name) or default).strip().lower()
    return v not in ("0", "false", "no", "off")


# Set PACT_USE_LOCAL_LLAMA_FOR_SYNTHESIS=0 to skip the LM and merge candidates in code
# (financial redacted string if present, else longest candidate, else original).
USE_LOCAL_LLAMA_FOR_SYNTHESIS = _env_truthy("PACT_USE_LOCAL_LLAMA_FOR_SYNTHESIS", "1")


def _local_llama_load_wait_timeout_sec() -> float | None:
    """
    How long /chat (and similar) will block waiting for the model.
    Env LOCAL_LLM_LOAD_TIMEOUT_SEC: seconds; 0 or negative means no limit; or
    unlimited/none/inf. Default 6 hours (first HF download often exceeds 45m).
    """
    raw = (os.environ.get("LOCAL_LLM_LOAD_TIMEOUT_SEC") or "").strip().lower()
    if raw in ("unlimited", "none", "inf", "infinite"):
        return None
    if raw == "":
        return 21600.0
    try:
        v = float(raw)
    except ValueError:
        return 21600.0
    if v <= 0:
        return None
    return max(60.0, v)


# Local Llama loading state (so we can expose progress to the frontend).
_local_llama_state_lock = threading.Lock()
_local_llama_loading = False
_local_llama_load_error: str | None = None
_local_llama_ready_event = threading.Event()


@app.on_event("startup")
async def startup_event():
    """Auto-load the Groq client on server startup (fast — just validates API key)."""
    global _local_llama_loading, _local_llama_load_error
    try:
        local_llama.load_model()
        local_llama.load_au_probe("")
        _local_llama_load_error = None
        _local_llama_ready_event.set()
        print("Groq backend initialised on startup.")
    except Exception as e:
        _local_llama_load_error = str(e)
        _local_llama_ready_event.set()  # unblock waiters so they get a proper error
        print(f"Startup load failed: {e}")


if local_llama.is_loaded():
    _local_llama_ready_event.set()

class LocalLlamaLoadRequest(BaseModel):
    model_name: str | None = None


def _background_load_local_llama(model_name: str) -> None:
    global _local_llama_loading, _local_llama_load_error
    try:
        local_llama.load_model(model_name=model_name)
        probe_path = os.path.abspath(os.path.join(ROOT_DIR, "data", "au_probe", "linearprobe_layer_32.pt"))
        local_llama.load_au_probe(probe_path, layer=32)
        
        with _local_llama_state_lock:
            _local_llama_load_error = None
            _local_llama_loading = False
            _local_llama_ready_event.set()
    except Exception as e:
        with _local_llama_state_lock:
            _local_llama_load_error = f"{type(e).__name__}: {str(e)[:300]}"
            _local_llama_loading = False
        # Set (not clear) so _ensure_local_llama_ready unblocks and can raise the error.
        _local_llama_ready_event.set()


def _ensure_local_llama_ready(timeout_sec: float | None = None) -> None:
    """
    Ensure local llama is loaded before running generation.
    If a background load is already running, wait for it.
    timeout_sec: override wait; None uses LOCAL_LLM_LOAD_TIMEOUT_SEC (default 6h).
    """
    global _local_llama_loading, _local_llama_load_error
    if local_llama.is_loaded():
        _local_llama_ready_event.set()
        return

    # Start a load if none is in progress.
    with _local_llama_state_lock:
        if not _local_llama_loading and not local_llama.is_loaded():
            _local_llama_loading = True
            _local_llama_load_error = None
            _local_llama_ready_event.clear()
            thread = threading.Thread(
                target=_background_load_local_llama,
                args=(LOCAL_LLM_MODEL_NAME,),
                daemon=True,
            )
            thread.start()

    effective = (
        timeout_sec if timeout_sec is not None else _local_llama_load_wait_timeout_sec()
    )
    if effective is None:
        ok = _local_llama_ready_event.wait()
    else:
        ok = _local_llama_ready_event.wait(timeout=effective)
    if not ok:
        raise HTTPException(
            status_code=503,
            detail=(
                "Local Llama load timed out while waiting for the model. "
                "First download can take a long time. Set LOCAL_LLM_LOAD_TIMEOUT_SEC "
                "to a larger value (seconds), 0 or unlimited for no limit, then restart "
                "the server. GET /local-llama/status shows load_error if startup failed."
            ),
        )

    if _local_llama_load_error:
        raise HTTPException(status_code=500, detail=f"Local Llama load failed: {_local_llama_load_error}")


@app.get("/local-llama/status")
async def local_llama_status():
    st = local_llama.get_status()
    with _local_llama_state_lock:
        st.update(
            {
                "loading": bool(_local_llama_loading),
                "load_error": _local_llama_load_error,
            }
        )
    return st


@app.post("/local-llama/load")
async def local_llama_load(req: LocalLlamaLoadRequest | None = None):
    global _local_llama_loading, _local_llama_load_error
    # Keep model_name optional so frontend can just "poke" the loader.
    model_name = LOCAL_LLM_MODEL_NAME
    if req and req.model_name:
        model_name = req.model_name

    with _local_llama_state_lock:
        if local_llama.is_loaded():
            _local_llama_ready_event.set()
            return {"started": False, "status": "ready", "model_name": model_name}
        if _local_llama_loading:
            return {"started": False, "status": "loading", "model_name": model_name}

        _local_llama_loading = True
        _local_llama_load_error = None
        _local_llama_ready_event.clear()
        thread = threading.Thread(
            target=_background_load_local_llama,
            args=(model_name,),
            daemon=True,
        )
        thread.start()

    return {"started": True, "status": "loading", "model_name": model_name}

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatSettings(BaseModel):
    identity: bool
    location: bool
    demographic: bool
    health: bool
    financial: bool

class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    query: str
    settings: ChatSettings
    api_key: str | None = None


class BatchChatRequest(BaseModel):
    queries: list[ChatRequest]
    api_key: str | None = None


class QueriesFile(BaseModel):
    version: int = 1
    queries: list[ChatRequest]


def _process_chat(request: ChatRequest) -> dict:
    original_query = request.query

    candidates, module_masks, financial_candidate = collect_pipeline_inputs(
        original_query, request.settings.model_dump()
    )

    final_prompt, llama_trace = _local_synthesize_final_prompt(
        original_query=original_query,
        candidates=candidates,
        privacy_preferences=request.settings.model_dump(),
        financial_candidate=financial_candidate,
    )

    au_score = 0.0
    if USE_LOCAL_LLAMA_FOR_SYNTHESIS:
        try:
            au_score = local_llama.get_au_uncertainty(final_prompt)
        except Exception as e:
            print(f"AU Uncertainty calculation failed: {e}")
            au_score = 0.0

    if au_score >= 0.8:
        response_text = (
            f"After redacting the information categories you selected, it looks like most of the "
            f"meaningful context has been removed from your query. Without enough detail, the AI is "
            f"likely to give you a vague or unhelpful response. Try rephrasing your query with more "
            f"context that you are comfortable sharing, or consider which privacy categories are "
            f"strictly necessary to redact. (Uncertainty score: {au_score:.2f})"
        )
    else:
        response_text = _cloud_llm(final_prompt, api_key=request.api_key)

    pipeline_trace = {
        "module_masks": module_masks,
        "candidates_for_llama": candidates,
        "privacy_preferences": request.settings.model_dump(),
        "local_llama": llama_trace,
        "au_probe": {
            "score": round(au_score, 4),
            "threshold": 0.8,
            "triggered": au_score >= 0.8,
            "status": "uncertain" if au_score >= 0.8 else "certain",
        },
        "final_prompt_to_gpt": final_prompt,
    }

    return {
        "response": response_text,
        "sanitizations": [],
        "original_query_sanitized": final_prompt,
        "pipeline_trace": pipeline_trace,
    }


def _escape_html(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


def _fallback_final_prompt(
    original_query: str,
    candidates: list[str],
    financial_candidate: str | None,
) -> str:
    if financial_candidate and financial_candidate.strip():
        return financial_candidate.strip()
    nonempty = [c.strip() for c in candidates if isinstance(c, str) and c.strip()]
    if nonempty:
        return max(nonempty, key=len)
    return original_query.strip() or original_query


def _local_synthesize_final_prompt(
    original_query: str,
    candidates: list[str],
    privacy_preferences: dict,
    financial_candidate: str | None = None,
) -> tuple[str, dict]:
    if len(candidates) == 1 and candidates[0].strip() == original_query.strip():
        trace = {
            "synthesis_mode": "bypass",
            "model_name": None,
            "synthesis_prompt": "",
            "raw_model_output": "",
            "extracted_before_fallback": original_query,
            "used_fallback": False,
            "fallback_reason": "no_masking_needed_or_all_toggles_off",
        }
        return original_query.strip(), trace

    prompt = build_privacy_synthesis_prompt(
        original_query, candidates, privacy_preferences
    )

    if not USE_LOCAL_LLAMA_FOR_SYNTHESIS:
        fp = _fallback_final_prompt(
            original_query, candidates, financial_candidate
        )
        trace = {
            "synthesis_mode": "deterministic_merge",
            "model_name": None,
            "synthesis_prompt": prompt,
            "raw_model_output": "",
            "extracted_before_fallback": fp,
            "used_fallback": False,
            "fallback_reason": None,
            "note": "Llama skipped (PACT_USE_LOCAL_LLAMA_FOR_SYNTHESIS=0). "
            "Policy: prefer financial redacted full line, else longest candidate, else original.",
        }
        return fp, trace

    _ensure_local_llama_ready()

    try:
        generated = local_llama.generate_text(
            prompt=prompt,
            max_new_tokens=128,
            temperature=0.2,
            top_p=0.9,
            model_name=LOCAL_LLM_MODEL_NAME,
            use_chat_template=True,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Local Llama generation failed: {type(e).__name__}: {str(e)[:300]}",
        ) from e

    raw = generated or ""
    synthesized = extract_final_prompt(raw)
    used_fallback = is_synthesis_unusable(synthesized)
    if used_fallback:
        final_prompt = _fallback_final_prompt(
            original_query, candidates, financial_candidate
        )
        fb_reason = "empty_or_meta_refusal"
    else:
        final_prompt = synthesized
        fb_reason = None

    trace = {
        "synthesis_mode": "local_llama",
        "model_name": LOCAL_LLM_MODEL_NAME,
        "synthesis_prompt": prompt,
        "raw_model_output": raw,
        "extracted_before_fallback": synthesized,
        "used_fallback": used_fallback,
        "fallback_reason": fb_reason,
    }
    return final_prompt, trace


def _cloud_llm(final_prompt: str, api_key: str | None = None) -> str:
    """
    Cloud GPT: chatbot-like response based only on final_prompt.
    Uses the per-request api_key if provided, falls back to the GPT_API_KEY env var.
    """
    key = (api_key or "").strip() or GPT_API_KEY
    if not key:
        raise HTTPException(
            status_code=400,
            detail="No OpenAI API key provided. Enter your key in the sidebar.",
        )
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": GPT_MODEL_ID,
        "messages": [{"role": "user", "content": final_prompt}],
        "temperature": 0.7,
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.HTTPError as e:
        status = getattr(resp, "status_code", "unknown")
        body = getattr(resp, "text", "")
        body = body[:300] + ("..." if len(body) > 300 else "")
        raise HTTPException(
            status_code=502,
            detail=f"GPT request failed (HTTP {status}): {body}",
        ) from e
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=502,
            detail=f"GPT request error: {type(e).__name__}: {str(e)[:300]}",
        ) from e

    content = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    content = content.strip() or "(No response from GPT.)"

    # UI uses innerHTML, so escape and convert newlines to <br>.
    return _escape_html(content).replace("\n", "<br>")


def _load_queries_document() -> dict:
    if not os.path.isfile(QUERIES_JSON_PATH):
        raise HTTPException(
            status_code=404,
            detail=f"Queries file not found: {QUERIES_JSON_PATH}",
        )
    with open(QUERIES_JSON_PATH, encoding="utf-8") as f:
        return json.load(f)


@app.get("/queries")
async def get_queries():
    """Return the query definitions from data/queries.json."""
    doc = _load_queries_document()
    QueriesFile.model_validate(doc)
    return doc


@app.post("/chat/batch")
async def chat_batch_endpoint(body: BatchChatRequest):
    """Run the same pipeline as /chat for each entry in the JSON body."""
    results = []
    for i, item in enumerate(body.queries):
        if body.api_key and not item.api_key:
            item.api_key = body.api_key
        out = _process_chat(item)
        out["index"] = i
        results.append(out)
    return {"count": len(results), "results": results}


@app.post("/chat/batch-from-file")
async def chat_batch_from_file(body: BatchChatRequest | None = None):
    """Load queries from data/queries.json and run the /chat pipeline for each."""
    doc = _load_queries_document()
    parsed = QueriesFile.model_validate(doc)
    results = []
    batch_key = body.api_key if body else None
    for i, item in enumerate(parsed.queries):
        if batch_key and not item.api_key:
            item.api_key = batch_key
        out = _process_chat(item)
        out["index"] = i
        results.append(out)
    return {"count": len(results), "results": results}


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    return _process_chat(request)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
