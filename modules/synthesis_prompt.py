"""
Shared prompt building + parsing for local Llama privacy synthesis.
Used by backend/server.py and scripts/test_local_llama.py (no GPT_API_KEY required).
"""

from __future__ import annotations

import json
import re


def build_privacy_synthesis_prompt(
    original_query: str,
    candidates: list[str],
    privacy_preferences: dict,
) -> str:
    """
    Candidates-first ordering: models latch onto the first block; treat that list as the
    authority for what wording (including placeholders) is allowed in the final line.
    """
    return (
        "You are a privacy-preserving text synthesizer. Your job is to rewrite the 'Original Query' "
        "by combining the 'Masked Variations' into a single coherent sentence.\n\n"
        "INSTRUCTIONS:\n"
        "1. Read the Original Query to understand the overall intent.\n"
        "2. Look at the Masked Variations. They contain the EXACT placeholders (like <PERSON>, <GIVEN_LOCATION>) "
        "or redactions you must use.\n"
        "3. Combine them into a single coherent output. If different variations use different labels for the same info (e.g., [REDACTED ID] vs [REDACTED ACCOUNT]), just use a single generic [REDACTED] tag to resolve the conflict.\n"
        "4. NOTE: ALL sensitive data (Identity, Financial, Health, etc.) has been ALREADY redacted by an upstream security layer. "
        "The placeholders you see ARE the finalized values. This task is for structural merging only and does NOT involve handling restricted info.\n"
        "5. DO NOT add any extra conversational text, labels, apologies, or markdown formatting.\n"
        "6. DO NOT summarize or truncate. DO NOT refuse to process based on 'safety'—this text is already sterilized. Output the full text with all redactions applied.\n\n"
        "EXAMPLE:\n"
        "Original Query: My name is John and I live in Paris.\n"
        "Masked Variations: [\"My name is <PERSON>\", \"I live in <GIVEN_LOCATION>\"]\n"
        "Output: My name is <PERSON> and I live in <GIVEN_LOCATION>.\n\n"
        "===\n\n"
        f"Masked Variations:\n{json.dumps(candidates, ensure_ascii=False)}\n\n"
        f"Original Query:\n{original_query}\n\n"
        f"Enabled Privacy Modules:\n{json.dumps(privacy_preferences, ensure_ascii=False)}\n\n"
        "Output:\n"
    )


def extract_final_prompt(hf_text: str) -> str:
    """
    Parse model output after privacy synthesis.

    The server prompt ends with ``final_prompt:`` (no trailing space), so the model
    often continues with the value only — no ``final_prompt:`` in the generated
    string. Models also ramble into ``` fences; we keep one line per spec.
    """
    txt = (hf_text or "").strip()
    if not txt:
        return ""
    # Never keep markdown / code the model was told not to emit.
    txt = txt.split("```")[0].strip()

    match = re.search(
        r"final_prompt\s*:\s*(.+?)(?:\r?\n|$)",
        txt,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match:
        inner = match.group(1).strip()
        if inner:
            return inner.splitlines()[0].strip()

    # Continuation-only: first non-empty line is the rewritten prompt.
    for line in txt.splitlines():
        s = line.strip()
        if s:
            return s
    return ""


_SYNTHESIS_UNUSABLE_RE = re.compile(
    r"(incomplete|cut off|truncat|could you (please )?clarif|provide (more )?context|"
    r"not sure what you|unclear|didn'?t understand|as an ai|i (can|cannot) '?t (help|provide|access|compromise|give guidance|guidance))",
    re.IGNORECASE,
)


def is_synthesis_unusable(text: str) -> bool:
    s = (text or "").strip()
    if not s:
        return True
    return bool(_SYNTHESIS_UNUSABLE_RE.search(s))
