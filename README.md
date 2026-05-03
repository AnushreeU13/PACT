# PACT: Privacy-Aware Cloud Transmission (Deployment Version)

PACT is a privacy middleware pipeline that intercepts user prompts before they reach a cloud AI, removes sensitive personal information, and only then forwards the sanitized version for a response. The user still receives a useful answer. The cloud never sees their real identity, health details, financial data, or location.

## The Problem

When a user types a prompt like "My name is Sarah Johnson, my SSN is 432-56-7890, I have Type 2 diabetes, can you help me fill this form?" the entire message goes to a cloud AI word for word. The user does not realize they have handed their most sensitive information to an external server. This happens constantly with medical questions, financial disclosures, insurance forms, and legal queries.

PACT sits between the user and the cloud LLM and makes sure that never happens.

## How It Works

1. The user submits a prompt through the PACT interface.
2. Five detection modules scan the prompt in parallel: identity, location, demographic, health, and financial.
3. Each detected entity is replaced with a tag such as `[REDACTED NAME]` or `[REDACTED HEALTH]`.
4. The AU-Probe estimates how uncertain the cloud LLM's response will be given how much was redacted. If the score exceeds the threshold, the prompt is flagged before being sent.
5. The sanitized prompt is forwarded to GPT-4o-mini.
6. The response is returned to the user.

---

## Project Structure

```
deployment_version/
├── backend/
│   └── server.py           # FastAPI server
├── frontend/
│   ├── index.html          # Main UI
│   ├── app.js              # Frontend logic
│   ├── style.css           # Styling
│   └── config.js           # Backend URL configuration
├── modules/
│   ├── local_llama.py      # Groq API wrapper + MiniLM AU-Probe
│   ├── identity_module.py  # Name, SSN, email, phone redaction
│   ├── modules_geo.py      # Address and location redaction
│   ├── demographic_module.py # Age, gender, race redaction
│   ├── health_module.py    # Medical entity redaction via Llama
│   ├── financial_detector.py # Card, account, and income redaction
│   ├── pipeline_collect.py # Module orchestration
│   ├── synthesis_prompt.py # Candidate merging logic
│   └── extract_docs.py     # PDF and image text extraction
├── data/
│   ├── au_probe/
│   │   └── minilm_probe.pt # Trained AU-Probe weights
│   └── queries.json        # Sample queries for testing
├── Procfile                # Railway start command
├── requirements.txt        # Python dependencies
└── .env.example            # Environment variable template
```

---

## Code Files

### `backend/server.py`

FastAPI server that exposes all PACT endpoints. At startup it loads the Groq client and the AU-Probe. It reads the `PORT` environment variable injected by Railway and binds to it. Key endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/status` | GET | Returns backend health and probe status |
| `/load` | POST | Loads the Groq model and AU-Probe |
| `/process` | POST | Runs the full PACT pipeline on a prompt |
| `/au_score` | GET | Returns the raw AU uncertainty score for a prompt |
| `/extract/text` | POST | Extracts plain text from an uploaded PDF or image |
| `/queries` | GET | Returns sample queries from `data/queries.json` |

---

### `modules/local_llama.py`

Wrapper for all LLM-related functionality in the deployment version. Replaces the Ollama backend used in the research version with two components:

**Text generation via Groq API**
Sends prompts to `llama-3.1-8b-instant` running on Groq's servers. Used by the health module and the synthesis step. Groq requires no local GPU and returns responses over HTTPS.

**AU-Probe via MiniLM**
Estimates prompt uncertainty using a lightweight sentence-transformer (`all-MiniLM-L6-v2`) that runs in-process on Railway. The probe applies:

```
score = sigmoid(w . embedding + b)
```

where `w` (shape [384]) and `b` (-3.8187) were learned by training logistic regression on MiniLM embeddings labeled by redaction ratio. A score close to 1 means the prompt is too degraded for a meaningful response. Falls back to a redaction-ratio heuristic if the probe file is not found.

---

### `modules/health_module.py`

Detects and redacts medical information using Llama directly as the detector. Instead of matching hardcoded patterns or using scispaCy, it sends a structured system prompt to Llama that defines four categories: CONDITION, MEDICATION, SYMPTOM, and PROCEDURE. Llama returns the redacted version of the input text. Temperature is set to 0 for deterministic output. Falls back gracefully if the Groq call fails.

---

### `modules/identity_module.py`

Detects and redacts personally identifiable information: full names, Social Security numbers, passport numbers, driver license numbers, email addresses, and phone numbers. Uses spaCy's `en_core_web_sm` model for named entity recognition combined with regex patterns for structured identifiers.

---

### `modules/modules_geo.py`

Detects and redacts geographic information: street addresses, city names, state and country references, and ZIP codes. Uses spaCy NER with geographic entity labels.

---

### `modules/demographic_module.py`

Detects and redacts demographic attributes: age, gender, race, ethnicity, and nationality. Uses a combination of spaCy NER and pattern matching for age expressions.

---

### `modules/financial_detector.py`

Detects and redacts financial information using a rule-based approach with structural validation:
- Credit and debit card numbers validated with the Luhn algorithm
- Bank account and routing numbers
- IBAN and SWIFT codes
- Cryptocurrency wallet addresses
- Income and salary references

---

### `modules/pipeline_collect.py`

Orchestrates all five modules. Runs them concurrently using `ThreadPoolExecutor` (one thread per module) and merges their outputs into a candidate list. After all modules complete, applies financial sanitization as a post-processing pass over the other modules' outputs to catch any card or account numbers that slipped through. Also contains `sequential_redaction_pipeline` for large documents where Llama synthesis is too slow.

---

### `modules/synthesis_prompt.py`

Builds the prompt sent to Llama that merges all module candidates into a single best-redacted version of the user's query. Handles cases where synthesis produces an unusable output and falls back to the longest candidate.

---

### `modules/extract_docs.py`

Extracts plain text from uploaded files. Uses PyMuPDF for PDF files and pytesseract (Tesseract OCR) for image files (PNG, JPG, TIFF). Returns the extracted text as a string for the pipeline to process.

---

### `frontend/config.js`

The only file that needs to change when deploying to a new Railway instance. Set `BACKEND_URL` to the Railway-provided HTTPS URL before pushing the frontend to GitHub Pages.

```js
const BACKEND_URL = "https://your-railway-app.up.railway.app";
```

---

## AU-Probe

### What It Is

The AU-Probe gates whether a redacted prompt should be sent to the cloud LLM. If too much context has been removed, the LLM cannot give a useful response. The probe detects this and flags the prompt instead of wasting an API call.

### How It Was Trained

A dataset of 488 prompts was generated from 40 seed prompts, each artificially redacted at ratios from 5% to 90% using PACT redaction tags. Labels were assigned using a redaction-ratio rule:

- Label 0 (certain): fewer than 30% of words are redaction tags
- Label 1 (uncertain): 30% or more of words are redaction tags

MiniLM embeddings (384-dimensional) were computed for each prompt. A logistic regression classifier was trained on these embeddings, learning a weight vector `w` and bias `b` that separates certain from uncertain prompts in embedding space.

**Training results:**
- Total prompts: 488 (303 certain, 185 uncertain)
- Train / test split: 390 / 98
- Accuracy: 0.93
- ROC-AUC: 0.9818

### Spot-Check Scores

| Prompt type | Score | Decision (threshold 0.8) |
|---|---|---|
| Clean question | 0.01 | Passes |
| Light redaction (~15%) | 0.05 | Passes |
| Medium redaction (~35%) | 0.46 | Passes |
| Heavy redaction (~70%) | 0.88 | Blocked |
| Fully redacted | 0.95 | Blocked |

### Retraining

```bash
python scripts/retrain_probe_minilm.py
```

Run from the repository root. The script saves updated weights to `data/au_probe/minilm_probe.pt`.

---

## Deployment

### Part 1: Frontend on GitHub Pages

The frontend is a static site and does not need a server.

1. Push the `frontend/` folder contents to a public GitHub repository.
2. Go to repository Settings and enable GitHub Pages from the root of the main branch.
3. Before pushing, update `frontend/config.js` with your Railway backend URL.

Your frontend will be live at `https://yourusername.github.io/your-repo/`.

---

### Part 2: Backend on Railway

Railway deploys directly from a GitHub repository and rebuilds automatically on every push.

**Prerequisites:**
- A Railway account at [railway.app](https://railway.app)
- A Groq API key from [console.groq.com](https://console.groq.com) (free tier available)
- This repository (or the `deployment_version/` folder) pushed to GitHub

**Steps:**

1. Create a new project on Railway and connect your GitHub repository.
2. If the repository contains more than just the deployment files, set the root directory to `deployment_version/` in Railway project settings.
3. Add the following environment variables in the Railway dashboard:

| Variable | Required | Value |
|---|---|---|
| `GROQ_API_KEY` | Yes | Your Groq API key |
| `GPT_API_KEY` | No | Your OpenAI API key (users can also enter this in the UI) |
| `GROQ_MODEL` | No | Override the Groq model. Defaults to `llama-3.1-8b-instant` |

4. Railway reads the `Procfile` and starts the server automatically:
   ```
   web: python backend/server.py
   ```
5. Railway injects a `PORT` environment variable. The server reads it and binds to the correct port.
6. Once deployed, copy the Railway-provided HTTPS URL into `frontend/config.js` and redeploy the frontend.

**Verifying the deployment:**

Visit `https://your-railway-app.up.railway.app/status`. You should see:

```json
{
  "loaded": true,
  "probe_ready": true,
  "backend": "groq",
  "groq_api_key_set": true
}
```

At startup the server logs:
```
AU probe: MiniLM probe loaded (w=torch.Size([384]), b=-3.8187).
Groq backend ready: model=llama-3.1-8b-instant
```

If `probe_ready` is false, check that `data/au_probe/minilm_probe.pt` is committed to the repository.

---

## Local Development

Requirements: Python 3.10+, a Groq API key.

```bash
cd deployment_version

pip install -r requirements.txt
python -m spacy download en_core_web_sm

cp .env.example .env
# Edit .env and set GROQ_API_KEY

python backend/server.py
```

Open `frontend/index.html` in a browser or serve it with any static file server.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes | Groq API key for Llama inference via Groq |
| `GPT_API_KEY` | No | OpenAI API key for GPT-4o-mini. Users can also provide this in the UI. |
| `GROQ_MODEL` | No | Groq model to use. Defaults to `llama-3.1-8b-instant`. |
| `PORT` | No | Injected automatically by Railway. Do not set manually. |

---

## Technologies

| Technology | Role |
|---|---|
| spaCy `en_core_web_sm` | Named entity recognition for identity, location, and demographic modules |
| Llama 3.1:8b-instant via Groq API | Text generation and health module detection |
| `all-MiniLM-L6-v2` (sentence-transformers) | Sentence embeddings for the AU-Probe |
| PyTorch | Linear probe weight storage and inference |
| scikit-learn | Logistic regression training for the AU-Probe |
| FastAPI + Uvicorn | REST API backend |
| PyMuPDF | PDF text extraction |
| pytesseract + Tesseract OCR | Image text extraction |
| Railway | Cloud backend hosting |
| GitHub Pages | Static frontend hosting |
| GPT-4o-mini (OpenAI) | Cloud LLM that receives the sanitized prompt |
