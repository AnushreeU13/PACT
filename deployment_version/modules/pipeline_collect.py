"""
Build privacy-module candidates for local Llama synthesis (same logic as the API).

Kept separate from backend/server.py so scripts can run without GPT_API_KEY.

Privacy guarantee: Groq (cloud Llama) is only called by the health module. To ensure
Groq never receives the original unredacted user query, all local modules (identity,
location, demographic, financial) run first and produce a pre-sanitized version of the
text. The health module receives only that pre-sanitized version.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Mapping

from modules.financial_detector import FinancialDetector
from modules import identity_module
from modules import demographic_module
from modules import health_module
from modules import modules_geo

_financial_detector = FinancialDetector()


def _pre_sanitize(text: str, settings: Mapping[str, Any]) -> str:
    """
    Apply all local (non-Groq) modules sequentially to produce a pre-sanitized string.
    This is passed to the health module so Groq never sees the original user query.
    """
    current = text
    if settings.get("financial"):
        current, _ = _financial_detector.detect_and_redact(current)
    if settings.get("identity"):
        current, _ = identity_module._get_detector().detect_and_redact(current)
    if settings.get("location"):
        current, _ = modules_geo._get_detector().detect_and_redact(current)
    if settings.get("demographic"):
        current, _ = demographic_module._get_detector().detect_and_redact(current)
    return current


def collect_pipeline_inputs(
    original_query: str,
    settings: Mapping[str, Any],
) -> tuple[list[str], dict[str, list[str]], str | None]:
    """
    Run enabled privacy modules; return merged candidates, per-module outputs, financial text.

    `settings` must include booleans: identity, location, demographic, health, financial.

    Local modules (identity, location, demographic, financial) run concurrently on the
    original query. The health module (Groq) runs afterward on a pre-sanitized version
    so that Groq never receives the original unredacted text.
    """
    module_masks: dict[str, list[str]] = {
        "identity": [],
        "location": [],
        "demographic": [],
        "health": [],
        "financial": [],
    }
    candidates: list[str] = []
    financial_candidate: str | None = None

    # Step 1: Run all local (non-Groq) modules concurrently on the original query.
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures: dict[str, Any] = {}
        if settings.get("identity"):
            futures["identity"] = ex.submit(
                identity_module.make_candidates_identity, original_query
            )
        if settings.get("location"):
            futures["location"] = ex.submit(
                modules_geo.make_candidates_location, original_query
            )
        if settings.get("demographic"):
            futures["demographic"] = ex.submit(
                demographic_module.make_candidates_demographic, original_query
            )
        if settings.get("financial"):
            futures["financial"] = ex.submit(
                _financial_detector.detect_and_redact, original_query
            )

        for name, fut in futures.items():
            res = fut.result()
            if name == "financial":
                processed_query, _ = res
                financial_candidate = processed_query
                module_masks["financial"] = [processed_query]
            else:
                module_masks[name] = [c for c in res if isinstance(c, str) and c.strip()]

    # If financial is enabled, sanitize other local candidates too (post-processing).
    if settings.get("financial"):
        for k in ("identity", "location", "demographic"):
            sanitized: list[str] = []
            for s in module_masks[k]:
                redacted, _ = _financial_detector.detect_and_redact(s)
                if isinstance(redacted, str) and redacted.strip():
                    sanitized.append(redacted)
            module_masks[k] = sanitized

    # Step 2: Run health module (Groq) on a pre-sanitized version of the query.
    # Local modules have already removed identity, location, demographic, and financial PII,
    # so Groq only ever sees text that has been partially redacted.
    if settings.get("health"):
        pre_sanitized = _pre_sanitize(original_query, settings)
        health_results = health_module.make_candidates_health(pre_sanitized)
        module_masks["health"] = [c for c in health_results if isinstance(c, str) and c.strip()]

    # Merge in stable order.
    for k in ("identity", "location", "demographic", "health", "financial"):
        candidates.extend(module_masks[k])

    candidates = [c for c in candidates if isinstance(c, str) and c.strip()]
    if not candidates:
        candidates = [financial_candidate.strip()] if financial_candidate else [original_query]

    return candidates, module_masks, financial_candidate


def sequential_redaction_pipeline(
    text: str,
    settings: Mapping[str, Any],
) -> str:
    """
    Apply enabled modules one-by-one to the same string.
    This is best for large documents where Llama synthesis is too slow.
    """
    current_text = text

    # Order matters: Financial and Identity typically cover the most ground
    if settings.get("financial"):
        current_text, _ = _financial_detector.detect_and_redact(current_text)
    
    if settings.get("identity"):
        current_text, _ = identity_module._get_detector().detect_and_redact(current_text)
        
    if settings.get("location"):
        current_text, _ = modules_geo._get_detector().detect_and_redact(current_text)
        
    if settings.get("demographic"):
        current_text, _ = demographic_module._get_detector().detect_and_redact(current_text)
        
    if settings.get("health"):
        current_text, _ = health_module._get_detector().detect_and_redact(current_text)

    return current_text
