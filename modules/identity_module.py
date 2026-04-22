from __future__ import annotations

import os
import re
from typing import Any

import spacy

_NLP = None
_DETECTOR = None

_EMAIL_RE = re.compile(
    r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[A-Za-z]{2,}\b"
)
_PHONE_RE = re.compile(
    r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}\b"
)
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
# Simple ID-like heuristic (not perfect): short alnum strings with mix.
_ID_RE = re.compile(r"\b(?:[A-Z]{1,2}\d{7,8}|\d{9,12})\b")


def _get_nlp():
    global _NLP
    if _NLP is not None:
        return _NLP

    try:
        _NLP = spacy.load("en_core_web_sm")
    except OSError:
        os.system("python -m spacy download en_core_web_sm")
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


class IdentityDetector:
    def __init__(self):
        self.nlp = _get_nlp()

    def detect_and_redact(self, text: str) -> tuple[str, list[dict[str, Any]]]:
        doc = self.nlp(text)

        # Collect all spans we want to redact.
        # Each entry: (start, end, replacement_text, original_text, type)
        spans: list[tuple[int, int, str, str, str]] = []

        for ent in doc.ents:
            if ent.label_ == "PERSON":
                spans.append((ent.start_char, ent.end_char, "[REDACTED NAME]", ent.text, "PERSON"))

        for m in _EMAIL_RE.finditer(text):
            spans.append((m.start(), m.end(), "[REDACTED EMAIL]", m.group(), "EMAIL_ADDRESS"))

        for m in _PHONE_RE.finditer(text):
            spans.append((m.start(), m.end(), "[REDACTED PHONE]", m.group(), "PHONE_NUMBER"))

        for m in _SSN_RE.finditer(text):
            spans.append((m.start(), m.end(), "[REDACTED ID]", m.group(), "SSN"))

        for m in _ID_RE.finditer(text):
            original = m.group()
            # Avoid clobbering card/account numbers here; those are financial-detector territory.
            if re.fullmatch(r"(?:\d[ -]?){13,19}", original):
                continue
            spans.append((m.start(), m.end(), "[REDACTED ID]", original, "ID"))

        # Non-overlap: prefer earlier spans.
        spans.sort(key=lambda x: (x[0], x[1] - x[0]))
        sanitized: list[dict[str, Any]] = []

        redacted_parts: list[str] = []
        last_idx = 0

        for start, end, replacement, original, typ in spans:
            if start < last_idx:
                continue
            redacted_parts.append(text[last_idx:start])
            redacted_parts.append(replacement)
            last_idx = end
            sanitized.append({"original": original, "type": typ})

        redacted_parts.append(text[last_idx:])
        return "".join(redacted_parts), sanitized


def _get_detector() -> IdentityDetector:
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = IdentityDetector()
    return _DETECTOR


def make_candidates_identity(text: str) -> list[str]:
    detector = _get_detector()
    redacted_text, _ = detector.detect_and_redact(text)
    return [redacted_text]

