from __future__ import annotations
import os
from typing import Any
from modules import local_llama

_DETECTOR = None

HEALTH_SYSTEM_PROMPT = (
    "You are an expert Medical identification-redaction Assistant. "
    "Your goal is to scan user text for health and medical information.\n\n"
    "Categories:\n"
    "- CONDITION: Diseases, disorders, or chronic illnesses.\n"
    "- MEDICATION: Drugs, brand names, generics, and their dosages (e.g., '10mg Prozac').\n"
    "- SYMPTOM: Physical or mental indicators (e.g., 'coughing', 'anxiety').\n"
    "- PROCEDURE: Tests, surgeries, or scans (e.g., 'MRI', 'blood work').\n\n"
    "Instructions:\n"
    "1. Detect and mask medical terms using the format [REDACTED CATEGORY] in the given user prompt.\n"
    "2. Preserve all other tokens, punctuation, and capitalization exactly as they appear.\n"
    "3. Respond ONLY with the final redacted prompt. Do NOT include any introductory text, apologies, or explanations.\n\n"
    "Example:\n"
    "- Input: 'I am John Doe, I have high palpitations recently. What should I do about it?'\n"
    "- Output: 'I am John Doe, I have high [REDACTED SYMPTOM] recently. What should I do about it?'"
)

class HealthDetector:
    """
    A 'zero-hardcode' health detector using a local Llama model via Ollama.
    """
    def detect_and_redact(self, text: str) -> tuple[str, list[dict[str, Any]]]:
        if not text.strip():
            return text, []

        prompt = f"{HEALTH_SYSTEM_PROMPT}\n\nInput User Prompt: '{text}'"
        
        try:
            # Call your local Ollama instance (configured in local_llama.py)
            redacted_text = local_llama.generate_text(
                prompt=prompt,
                max_new_tokens=len(text) + 100, # Allow enough space for the tags
                temperature=0.0, # Zero temperature for consistent redaction
                top_p=0.9,
                use_chat_template=True
            )
            
            # Clean up potential LLM chatter (stripping extra quotes or whitespace)
            redacted_text = redacted_text.strip().strip("'").strip('"')
            
            # If the LLM returns an empty string or fails, fallback to the original text
            if not redacted_text:
                return text, []
            
            # We track the 'redaction event' for the logs
            # Since Llama is doing the work, we don't have a list of individual spans,
            # but we can log that a LLM-based health redaction occurred.
            sanitizations = []
            if redacted_text != text:
                sanitizations.append({"original": "LLM-identified medical context", "type": "HEALTH_MODEL_Ollama"})
            
            return redacted_text, sanitizations

        except Exception as e:
            # Fail gracefully: if the LLM is down, return the original text
            # This prevents the whole PACT pipeline from crashing.
            print(f"Health Module Error (Llama): {e}")
            return text, []

def _get_detector() -> HealthDetector:
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = HealthDetector()
    return _DETECTOR

def make_candidates_health(text: str) -> list[str]:
    """
    Called by the PACT pipeline.
    """
    detector = _get_detector()
    redacted_text, _ = detector.detect_and_redact(text)
    return [redacted_text]
