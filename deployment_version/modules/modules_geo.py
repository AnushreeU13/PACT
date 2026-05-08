from __future__ import annotations

import os
import re
from typing import Any

import spacy

_NLP = None
_DETECTOR = None

_US_STATE_NAMES = {
    "alabama",
    "alaska",
    "arizona",
    "arkansas",
    "california",
    "colorado",
    "connecticut",
    "delaware",
    "florida",
    "georgia",
    "hawaii",
    "idaho",
    "illinois",
    "indiana",
    "iowa",
    "kansas",
    "kentucky",
    "louisiana",
    "maine",
    "maryland",
    "massachusetts",
    "michigan",
    "minnesota",
    "mississippi",
    "missouri",
    "montana",
    "nebraska",
    "nevada",
    "new hampshire",
    "new jersey",
    "new mexico",
    "new york",
    "north carolina",
    "north dakota",
    "ohio",
    "oklahoma",
    "oregon",
    "pennsylvania",
    "rhode island",
    "south carolina",
    "south dakota",
    "tennessee",
    "texas",
    "utah",
    "vermont",
    "virginia",
    "washington",
    "west virginia",
    "wisconsin",
    "wyoming",
}

_US_STATE_ABBR = {
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
    "DC",
}

_COUNTRIES = {
    "india",
    "china",
    "korea",
    "japan",
    "germany",
    "france",
    "united states",
    "usa",
    "canada",
    "united kingdom",
    "uk",
    "australia",
    "brazil",
    "mexico",
    "spain",
    "italy",
    "netherlands",
    "sweden",
    "norway",
    "denmark",
    "finland",
    "switzerland",
    "russia",
    "turkey",
}

_ADDRESS_RE = re.compile(
    r"\b\d{1,6}\s+[A-Za-z0-9.\- ]{1,40}\s+(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Ln|Lane|Dr|Drive|Way|Ter|Terrace)\b",
    flags=re.IGNORECASE,
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


def _generalize_location(ent_text: str) -> tuple[str, str]:
    """
    Returns (replacement_text, type).
    """
    raw = ent_text.strip()
    low = raw.lower()

    if raw.upper() in _US_STATE_ABBR:
        return "a U.S. state", "US_STATE"

    if low in _US_STATE_NAMES:
        return "a U.S. state", "US_STATE"

    if low in _COUNTRIES or low.startswith("the ") and low[4:] in _COUNTRIES:
        return "a country", "COUNTRY"

    # Default for GPE/LOC: coarse location
    return "[REDACTED LOCATION]", "LOCATION"


class GeoDetector:
    def __init__(self):
        self.nlp = _get_nlp()

    def detect_and_redact(self, text: str) -> tuple[str, list[dict[str, Any]]]:
        doc = self.nlp(text)
        spans: list[tuple[int, int, str, str, str]] = []

        # Address-like patterns
        for m in _ADDRESS_RE.finditer(text):
            original = m.group()
            spans.append((m.start(), m.end(), "[REDACTED LOCATION]", original, "ADDRESS"))

        for ent in doc.ents:
            if ent.label_ not in ("GPE", "LOC"):
                continue

            replacement, typ = _generalize_location(ent.text)
            # Spec preference: Chicago -> a U.S. city.
            # If we didn't classify as state/country, use U.S. city as a coarse generalization.
            if typ == "LOCATION":
                replacement = "a U.S. city"

            spans.append((ent.start_char, ent.end_char, replacement, ent.text, typ))

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


def _get_detector() -> GeoDetector:
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = GeoDetector()
    return _DETECTOR


def make_candidates_location(text: str) -> list[str]:
    detector = _get_detector()
    redacted_text, _ = detector.detect_and_redact(text)
    return [redacted_text]
