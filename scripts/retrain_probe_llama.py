"""
Retrain the AU-Probe linear classifier using Llama 3.1:8b embeddings from Ollama.

Labels come from redaction density (same rule as retrain_probe_minilm.py):
  - ratio = count([REDACTED...] tokens) / total words
  - label = 1 (uncertain) if ratio >= UNCERTAIN_THRESHOLD else 0 (certain)

Ollama must be running with llama3.1:8b available:
    ollama serve
    ollama pull llama3.1:8b

Run from repo root:
    python scripts/retrain_probe_llama.py

Saves updated weights to:  data/au_probe/linearprobe_layer_32.pt
"""

from __future__ import annotations

import os
import random
import re
import time

import numpy as np
import requests
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT            = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_PROBE    = os.path.join(ROOT, "data", "au_probe", "linearprobe_layer_32.pt")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.environ.get("LOCAL_LLM_MODEL_NAME", "llama3.1:8b")

UNCERTAIN_THRESHOLD = 0.30
REDACTED_RE = re.compile(r'\[REDACTED[^\]]*\]', re.IGNORECASE)

# ---------------------------------------------------------------------------
# Prompt bank (same as MiniLM script for consistency)
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def redact_at_ratio(text: str, target_ratio: float, rng: random.Random) -> str:
    words = text.split()
    n = len(words)
    n_redact = max(0, min(int(round(target_ratio * n)), n))
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
    prompts, labels = [], []

    def add(text: str) -> None:
        actual = compute_ratio(text)
        prompts.append(text)
        labels.append(1 if actual >= UNCERTAIN_THRESHOLD else 0)

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


def get_ollama_embedding(text: str, retries: int = 3) -> list[float] | None:
    for attempt in range(retries):
        try:
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                json={"model": OLLAMA_MODEL, "prompt": text},
                timeout=60.0,
            )
            resp.raise_for_status()
            embedding = resp.json().get("embedding")
            if embedding:
                return embedding
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2.0)
            else:
                print(f"\n  WARNING: Failed to get embedding after {retries} attempts: {e}")
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== Step 1: Checking Ollama connectivity ===")
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
        resp.raise_for_status()
        models = [m.get("name", "") for m in resp.json().get("models", [])]
        print(f"  Ollama reachable. Available models: {models}")
        if not any(m.startswith(OLLAMA_MODEL.split(":")[0]) for m in models):
            print(f"  ERROR: Model '{OLLAMA_MODEL}' not found. Run: ollama pull {OLLAMA_MODEL}")
            return
        print(f"  Model '{OLLAMA_MODEL}' confirmed available.")
    except Exception as e:
        print(f"  ERROR: Cannot reach Ollama at {OLLAMA_BASE_URL}: {e}")
        print("  Make sure Ollama is running: ollama serve")
        return

    print("\n=== Step 2: Building dataset with redaction-ratio labels ===")
    rng = random.Random(42)
    prompts, labels = build_dataset(rng)
    n_certain   = labels.count(0)
    n_uncertain = labels.count(1)
    print(f"  Total: {len(prompts)} prompts")
    print(f"  Certain (0): {n_certain}  |  Uncertain (1): {n_uncertain}")

    print(f"\n=== Step 3: Fetching Llama embeddings from Ollama ===")
    print(f"  Model: {OLLAMA_MODEL}  |  This will take several minutes...")
    embeddings = []
    valid_prompts, valid_labels = [], []
    t0 = time.time()

    for i, (text, label) in enumerate(zip(prompts, labels)):
        emb = get_ollama_embedding(text)
        if emb is not None:
            embeddings.append(emb)
            valid_prompts.append(text)
            valid_labels.append(label)
        elapsed = time.time() - t0
        avg = elapsed / (i + 1)
        remaining = avg * (len(prompts) - i - 1)
        print(
            f"  [{i+1}/{len(prompts)}] dim={len(emb) if emb else 'ERR'}"
            f"  elapsed={elapsed:.0f}s  est_remaining={remaining:.0f}s",
            end="\r",
        )

    print(f"\n  Done. Got {len(embeddings)}/{len(prompts)} embeddings.")
    if len(embeddings) < 50:
        print("  ERROR: Too few embeddings collected. Check Ollama and retry.")
        return

    emb_dim = len(embeddings[0])
    print(f"  Embedding dimension: {emb_dim}")

    print(f"\n=== Step 4: Training logistic regression ===")
    X = np.array(embeddings, dtype=np.float32)
    y = np.array(valid_labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train)}  Test: {len(X_test)}")

    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    print("\n  Classification report:")
    print(classification_report(y_test, y_pred, target_names=["certain", "uncertain"]))
    auc = roc_auc_score(y_test, y_prob)
    print(f"  ROC-AUC: {auc:.4f}")

    print(f"\n=== Step 5: Spot-checking scores ===")
    w_t = torch.tensor(clf.coef_[0], dtype=torch.float32)
    b_t = float(clf.intercept_[0])

    test_cases = [
        ("Clean - general question",
         "What is the best way to learn Python programming?"),
        ("Clean - PII present (unredacted)",
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

    import math
    for label, text in test_cases:
        emb = get_ollama_embedding(text)
        if emb is None:
            print(f"  [{label}]  SKIPPED (embedding failed)")
            continue
        emb_t = torch.tensor(emb, dtype=torch.float32)
        if emb_t.shape[0] != w_t.shape[0]:
            emb_t = emb_t[:w_t.shape[0]]
        logit = torch.dot(w_t, emb_t).item() + b_t
        score = 1.0 / (1.0 + math.exp(-logit))
        ratio = compute_ratio(text)
        print(f"  [{label}]")
        print(f"    ratio={ratio:.2f}  score={score:.4f}  {'UNCERTAIN (blocked)' if score >= 0.5 else 'CERTAIN (passes)'}")

    print(f"\n=== Step 6: Saving probe ===")
    probe_data = {
        "w":          w_t,
        "b":          b_t,
        "layer":      32,
        "backbone":   OLLAMA_MODEL,
        "label_rule": f"uncertain if redaction_ratio >= {UNCERTAIN_THRESHOLD}",
        "best_lambda": 1.0 / clf.C,
        "n_train":    len(X_train),
        "n_test":     len(X_test),
        "roc_auc":    round(auc, 4),
        "emb_dim":    emb_dim,
    }
    os.makedirs(os.path.dirname(OUTPUT_PROBE), exist_ok=True)
    torch.save(probe_data, OUTPUT_PROBE)
    print(f"  Saved: {OUTPUT_PROBE}")
    print(f"  w shape: {w_t.shape}  b: {b_t:.4f}  emb_dim: {emb_dim}")
    print(f"  ROC-AUC: {auc:.4f}")
    print("\nDone.")


if __name__ == "__main__":
    main()
