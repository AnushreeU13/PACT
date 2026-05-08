"""
Retrain the AU-Probe using logistic regression on sentence-transformer embeddings.

Labels come from redaction density:
  - ratio = count([REDACTED...] tokens) / total words
  - label = 1 (uncertain) if ratio >= UNCERTAIN_THRESHOLD else 0 (certain)

The probe outputs sigmoid(w . embedding + b) as the AU score.
The user-set threshold in the UI is compared against this score at inference time.

Run from repo root:
    python scripts/retrain_probe_minilm.py
"""

from __future__ import annotations

import os
import random
import re

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT               = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_PROBE       = os.path.join(ROOT, "data", "au_probe", "minilm_probe.pt")
ST_MODEL           = "all-MiniLM-L6-v2"
REDACTED_RE        = re.compile(r'\[REDACTED[^\]]*\]', re.IGNORECASE)
UNCERTAIN_THRESHOLD = 0.30

# ---------------------------------------------------------------------------
# Prompt bank
# ---------------------------------------------------------------------------
CLEAN_PROMPTS = [
    "What is the capital of France?",
    "Explain how photosynthesis works in simple terms.",
    "What are the best practices for writing clean Python code?",
    "How does compound interest work over 30 years?",
    "What is the difference between machine learning and deep learning?",
    "Can you recommend a good book on data privacy?",
    "How do I improve my public speaking skills?",
    "What causes inflation in an economy?",
    "Explain the concept of a relational database.",
    "What are the benefits of regular exercise?",
    "How does the human immune system fight viruses?",
    "What is the difference between a REST API and a GraphQL API?",
    "How do solar panels convert sunlight to electricity?",
    "What is the significance of the Turing test?",
    "How does encryption protect data in transit?",
    "My name is John Smith and I live at 123 Oak Street, Chicago IL. I need help finding a doctor.",
    "My SSN is 432-56-7890 and my date of birth is March 14 1985. Can you help me fill out this form?",
    "I am Sarah Johnson, email sjohnson@gmail.com, phone 312-555-0192. I need to reset my account.",
    "My credit card number is 4111-1111-1111-1111, expiry 09/27, CVV 123. Is there a better rewards card?",
    "My bank account is 7734829100 at routing 064000017. How do I set up direct deposit?",
    "I was diagnosed with Type 2 diabetes last month. My HbA1c is 8.2 and I take metformin 500mg.",
    "Patient Michael Chen, DOB 1975-06-22, insurance ID HMO-98765, presents with stage 2 hypertension.",
    "I live at 55 River Lane, Boston MA 02134 and work at TechCorp Inc on 200 State Street.",
    "My passport number is P123456789 and I travel frequently between New York and London.",
    "As a 45-year-old Hispanic woman, I am applying for a senior engineering role at this company.",
    "I am John Doe, aged 52, and I was recently diagnosed with chronic kidney disease. What diet should I follow?",
    "My daughter Emma, age 8, has a peanut allergy. What foods should she avoid at school?",
    "I live in Seattle and my annual income is $85,000. Am I eligible for this tax credit?",
    "My name is Alice Wong and I take sertraline 100mg daily. Are there any food interactions?",
    "I am a 30-year-old male in Austin Texas with a BMI of 28. What health screenings should I get?",
    "How do I negotiate a salary increase at my current job?",
    "What is the best way to save for retirement in my 30s?",
    "Explain the difference between term life and whole life insurance.",
    "What are the symptoms of a vitamin D deficiency?",
    "How does the stock market work for a complete beginner?",
    "Can you summarize the key provisions of GDPR?",
    "What is the difference between a will and a living trust?",
    "How should I prepare for a job interview at a tech company?",
    "What are some effective study techniques for graduate students?",
    "Explain how neural networks learn from data.",
]

REDACTION_TAGS = [
    "[REDACTED NAME]",
    "[REDACTED LOCATION]",
    "[REDACTED ID]",
    "[REDACTED FINANCIAL]",
    "[REDACTED HEALTH]",
    "[REDACTED DATE]",
    "[REDACTED]",
]


def redact_at_ratio(text: str, target_ratio: float, rng: random.Random) -> str:
    words = text.split()
    n = len(words)
    n_redact = int(round(target_ratio * n))
    n_redact = max(0, min(n_redact, n))
    if n_redact == 0:
        return text
    indices = rng.sample(range(n), n_redact)
    for idx in indices:
        words[idx] = rng.choice(REDACTION_TAGS)
    return " ".join(words)


def compute_ratio(text: str) -> float:
    words = text.split()
    if not words:
        return 0.0
    return len(REDACTED_RE.findall(text)) / len(words)


def build_dataset(rng: random.Random) -> tuple[list[str], list[int]]:
    prompts: list[str] = []
    labels: list[int] = []

    def add(text: str) -> None:
        prompts.append(text)
        ratio = compute_ratio(text)
        labels.append(1 if ratio >= UNCERTAIN_THRESHOLD else 0)

    for p in CLEAN_PROMPTS:
        add(p)

    for p in CLEAN_PROMPTS:
        for ratio in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.75, 0.90]:
            add(redact_at_ratio(p, ratio, rng))

    heavy_templates = [
        "[REDACTED NAME] [REDACTED LOCATION] [REDACTED ID] [REDACTED FINANCIAL] [REDACTED HEALTH] [REDACTED DATE].",
        "My [REDACTED NAME] is [REDACTED ID] at [REDACTED LOCATION] with [REDACTED FINANCIAL] and [REDACTED HEALTH].",
        "[REDACTED] [REDACTED] [REDACTED] [REDACTED] [REDACTED] [REDACTED] [REDACTED].",
        "I am [REDACTED NAME] and I have [REDACTED HEALTH] at [REDACTED LOCATION] since [REDACTED DATE].",
        "[REDACTED NAME] [REDACTED NAME] [REDACTED LOCATION] [REDACTED FINANCIAL] [REDACTED ID].",
        "Can you help [REDACTED NAME] at [REDACTED LOCATION] with [REDACTED HEALTH] and [REDACTED FINANCIAL]?",
        "The [REDACTED NAME] at [REDACTED LOCATION] needs [REDACTED HEALTH] by [REDACTED DATE] costing [REDACTED FINANCIAL].",
        "[REDACTED ID] [REDACTED NAME] [REDACTED DATE] [REDACTED LOCATION] [REDACTED FINANCIAL] [REDACTED HEALTH].",
    ]
    for t in heavy_templates:
        add(t)

    return prompts, labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    rng = random.Random(42)

    print("=== Step 1: Building dataset (binary labels: uncertain if ratio >= 0.30) ===")
    prompts, labels = build_dataset(rng)
    y_arr = np.array(labels)
    print(f"  Total: {len(prompts)} prompts")
    print(f"  Certain (0): {(y_arr == 0).sum()}  Uncertain (1): {(y_arr == 1).sum()}")

    print(f"\n=== Step 2: Getting sentence-transformer embeddings ===")
    st_model = SentenceTransformer(ST_MODEL)
    embeddings = st_model.encode(prompts, show_progress_bar=True, normalize_embeddings=True)
    print(f"  Embeddings shape: {embeddings.shape}")

    print(f"\n=== Step 3: Training logistic regression (binary labels) ===")
    X = embeddings
    y = np.array(labels, dtype=np.int32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train)}  Test: {len(X_test)}")

    clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"  Accuracy: {accuracy:.4f}  ROC-AUC: {roc_auc:.4f}")

    print(f"\n=== Step 4: Spot-checking scores (sigmoid of logit) ===")
    test_cases = [
        ("Clean - general question",
         "What is the best way to learn Python programming?"),
        ("Clean - PII present (not yet redacted)",
         "My name is Alice and I live in Boston. What doctor should I see?"),
        ("Light redaction (~15%)",
         "My name is [REDACTED NAME] and I live in Boston. What doctor should I see?"),
        ("Medium redaction (~35%)",
         "My [REDACTED NAME] is [REDACTED ID] and I live at [REDACTED LOCATION]. What should I do?"),
        ("Heavy redaction (~70%)",
         "[REDACTED NAME] [REDACTED LOCATION] [REDACTED ID] [REDACTED FINANCIAL] help me with [REDACTED HEALTH]."),
        ("Fully redacted",
         "[REDACTED NAME] [REDACTED LOCATION] [REDACTED ID] [REDACTED FINANCIAL] [REDACTED HEALTH] [REDACTED DATE]."),
    ]
    w_t = torch.tensor(clf.coef_, dtype=torch.float32).squeeze()
    b_t = float(clf.intercept_[0])
    for label, text in test_cases:
        emb = st_model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        emb_t = torch.tensor(emb, dtype=torch.float32)
        logit = torch.dot(w_t, emb_t).item() + b_t
        score = float(torch.sigmoid(torch.tensor(logit)).item())
        ratio = compute_ratio(text)
        print(f"  [{label}]")
        print(f"    actual_ratio={ratio:.2f}  sigmoid_score={score:.4f}")

    print(f"\n=== Step 5: Saving probe ===")
    probe_data = {
        "w":          w_t,
        "b":          b_t,
        "layer":      0,
        "backbone":   ST_MODEL,
        "target":     "binary (uncertain if redaction_ratio >= 0.30)",
        "model_type": "LogisticRegression",
        "n_train":    len(X_train),
        "n_test":     len(X_test),
        "accuracy":   round(float(accuracy), 4),
        "roc_auc":    round(float(roc_auc), 4),
    }
    os.makedirs(os.path.dirname(OUTPUT_PROBE), exist_ok=True)
    torch.save(probe_data, OUTPUT_PROBE)
    print(f"  Saved: {OUTPUT_PROBE}")
    print(f"  w shape: {w_t.shape}, b: {b_t:.4f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
