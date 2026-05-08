from __future__ import annotations

import os
import re
from typing import Any

import spacy
from spacy.pipeline import EntityRuler

_NLP = None
_DETECTOR = None

_NATIONALITY_WORDS = {
    "indian",
    "korean",
    "chinese",
    "japanese",
    "american",
    "mexican",
    "brazilian",
    "canadian",
    "british",
    "german",
    "french",
    "italian",
    "spanish",
    "turkish",
    "russian",
    "vietnamese",
    "thai",
    "filipino",
    "nigerian",
    "pakistani",
    "bangladeshi",
}

_STUDENT_NATIONALITY_RE = re.compile(
    r"\b(?P<nationality>" + "|".join(sorted(_NATIONALITY_WORDS)) + r")\s+student(s)?\b",
    flags=re.IGNORECASE,
)

_KOREAN_AMERICAN_RE = re.compile(r"\bKorean[- ]American\b", flags=re.IGNORECASE)
_INTERNATIONAL_STUDENT_RE = re.compile(r"\binternational student(s)?\b", flags=re.IGNORECASE)
_FIRST_GEN_IMMIGRANT_RE = re.compile(
    r"\bfirst[- ]generation immigrant(s)?\b", flags=re.IGNORECASE
)


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


class DemographicDetector:
    def __init__(self):
        self.nlp = _get_nlp()
        self._setup_ruler()

    def _setup_ruler(self) -> None:
        # Avoid duplicate pipes if detector is created more than once.
        if any(pn == "demographic_ruler" for pn in self.nlp.pipe_names):
            return

        ruler = self.nlp.add_pipe("entity_ruler", name="demographic_ruler", after="ner")
        assert isinstance(ruler, EntityRuler)

        patterns = [
            {"label": "DEMOGRAPHIC", "pattern": "international student"},
            {"label": "DEMOGRAPHIC", "pattern": "immigrant family"},
            {"label": "DEMOGRAPHIC", "pattern": "first-generation immigrant"},
            {"label": "DEMOGRAPHIC", "pattern": "Korean-American"},
            {"label": "DEMOGRAPHIC", "pattern": "passport holder"},
        ]
        ruler.add_patterns(patterns)

    def detect_and_redact(self, text: str) -> tuple[str, list[dict[str, Any]]]:
        doc = self.nlp(text)

        spans: list[tuple[int, int, str, str, str]] = []

        # Regex-driven generalizations (fast + predictable)
        for m in _STUDENT_NATIONALITY_RE.finditer(text):
            spans.append((m.start(), m.end(), "international student", m.group(), "DEMOGRAPHIC"))

        for m in _INTERNATIONAL_STUDENT_RE.finditer(text):
            spans.append((m.start(), m.end(), "international student", m.group(), "DEMOGRAPHIC"))

        for m in _KOREAN_AMERICAN_RE.finditer(text):
            spans.append((m.start(), m.end(), "someone from a multicultural background", m.group(), "DEMOGRAPHIC"))

        for m in _FIRST_GEN_IMMIGRANT_RE.finditer(text):
            spans.append((m.start(), m.end(), "someone from a first-generation immigrant background", m.group(), "DEMOGRAPHIC"))

        # spaCy NORP entities
        for ent in doc.ents:
            if ent.label_ == "NORP":
                replacement = "[REDACTED DEMOGRAPHIC]"
                low = ent.text.lower()
                if low in _NATIONALITY_WORDS:
                    # Match example: "Indian student" => "international student"
                    replacement = "someone from a multicultural background"
                spans.append((ent.start_char, ent.end_char, replacement, ent.text, "NORP"))
            elif ent.label_ == "DEMOGRAPHIC":
                # Patterns we defined in EntityRuler
                low = ent.text.lower()
                if "international student" in low:
                    replacement = "international student"
                elif "korean-american" in low:
                    replacement = "someone from a multicultural background"
                elif "first-generation immigrant" in low:
                    replacement = "someone from a first-generation immigrant background"
                else:
                    replacement = "[REDACTED DEMOGRAPHIC]"
                spans.append((ent.start_char, ent.end_char, replacement, ent.text, "DEMOGRAPHIC"))

        # Prefer longer spans when they start at the same character.
        spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))
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


def _get_detector() -> DemographicDetector:
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = DemographicDetector()
    return _DETECTOR


def make_candidates_demographic(text: str) -> list[str]:
    detector = _get_detector()
    redacted_text, _ = detector.detect_and_redact(text)
    return [redacted_text]

