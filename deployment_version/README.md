# PACT Deployment Version

This folder contains a cloud-deployable version of PACT built for demonstration and online access. It is **not** the primary research version. The local version (in the repository root) runs entirely on-device using Ollama. This version replaces Ollama with Groq's hosted Llama API so that PACT can run on a server without requiring a GPU or a local model download. The backend runs on Railway; the frontend is served from GitHub Pages.

---

## Why a Deployment Version Exists

The local version requires the user to have Ollama installed and a 4.7 GB model pulled — which is not practical for a live demo or for reviewers who just want to see PACT in action. This version removes that barrier. Anyone with a browser can reach the frontend and interact with PACT without any local setup.

The tradeoff is that text generation and health module detection now go to Groq's servers instead of the user's machine. This is a deliberate compromise accepted for demo purposes only.

---

## What Changed from the Local Version

### 1. LLM Backend: Ollama → Groq API

The local version talks to a locally running Ollama instance at `http://localhost:11434`. This version calls Groq's API (`api.groq.com`) using the `llama-3.1-8b-instant` model. Groq requires no local GPU and returns responses over HTTPS with low latency.

**Affected file:** `modules/local_llama.py` — replaced the Ollama HTTP client with the `groq` Python SDK.

### 2. AU-Probe: Llama Embeddings → MiniLM Embeddings

The local version fetches 4096-d embeddings from Ollama to score prompts. Groq does not expose an embeddings endpoint, so the probe was retrained on `all-MiniLM-L6-v2` (384-d, runs in-process via `sentence-transformers`). The linear classifier was retrained on the same 488-prompt dataset using redaction-ratio labels.

**Affected file:** `modules/local_llama.py` — probe now loads MiniLM weights and uses `sentence_transformers.SentenceTransformer` instead of Ollama.

**Probe training results (MiniLM):**
- Accuracy: 0.93 — ROC-AUC: 0.9818

### 3. Synthesis Prompt: Original Query Removed

In an early version of the pipeline, the synthesis prompt sent to Groq Llama included the user's original unredacted query alongside the redacted candidates. This meant Groq could see the raw PII.

This was corrected: `modules/synthesis_prompt.py` now builds a prompt that contains **only the redacted candidates** and the enabled module list. The original query is never included. Groq's Llama only sees text that has already been partially sanitized by the local modules.

**Affected file:** `modules/synthesis_prompt.py`

### 4. Health Module Still Uses Groq Llama

The health module sends text to Groq Llama (temperature 0) to detect medical entities. There was no local alternative that matched its recall. Because of the synthesis prompt fix above, the health module receives a pre-sanitized version of the query (identity, location, demographic, and financial PII have already been removed by local modules before health is called), but it still operates on cloud infrastructure.

**Remaining limitation:** The local modules (identity, location, demographic, financial) each produce a partially redacted version of the original query. By the time health is called, some PII is gone, but not all — the other modules each only redact their own category. A query with a person's name, address, and a medical condition would have the name and address stripped before health sees it, but each local module's output still contains the other categories' raw data. The union of all candidates passed to synthesis still contains partially unredacted text. The synthesis prompt fix ensures Groq never sees the *original* query directly, but it does see the candidates produced by the local modules — which are partially redacted, not fully clean.

This is a known and accepted limitation of the current design. A complete mitigation would require running all five modules sequentially and passing only the fully chained output to Groq — which is what `sequential_redaction_pipeline` in `pipeline_collect.py` does, but that path is used only for large documents, not for the standard query flow.

### 5. Infrastructure

| Component | Local version | Deployment version |
|---|---|---|
| Backend | `uvicorn` on localhost | Railway (reads `PORT` env var) |
| Frontend | Static file opened in browser | GitHub Pages |
| Config | Environment variables or `.env` | Railway dashboard env vars |
| Start command | `python backend/server.py` | `Procfile`: `web: python backend/server.py` |

---

## Folder Structure

```
deployment_version/
├── backend/
│   └── server.py           # FastAPI server (Railway-compatible)
├── frontend/
│   ├── index.html          # Main UI
│   ├── app.js              # Frontend logic
│   ├── style.css           # Styling
│   └── config.js           # Set BACKEND_URL here before deploying
├── modules/
│   ├── local_llama.py      # Groq API wrapper + MiniLM AU-Probe
│   ├── pipeline_collect.py # Module orchestration
│   ├── synthesis_prompt.py # Candidate merging (original query excluded)
│   ├── identity_module.py
│   ├── modules_geo.py
│   ├── demographic_module.py
│   ├── health_module.py    # Medical detection via Groq Llama
│   ├── financial_detector.py
│   └── extract_docs.py
├── data/
│   └── au_probe/
│       └── minilm_probe.pt # MiniLM probe weights (384-d)
├── Procfile                # Railway start command
├── requirements.txt
├── .env.example
└── README.md
```

---

## Deploying

### Part 1 — Backend on Railway

Railway builds and runs the backend directly from a GitHub repository.

**Prerequisites:**
- A Railway account at [railway.app](https://railway.app)
- A Groq API key from [console.groq.com](https://console.groq.com) (free tier is sufficient)
- This repository pushed to GitHub

**Steps:**

1. Go to [railway.app](https://railway.app), create a new project, and select **Deploy from GitHub repo**.
2. Select this repository and set the **Root Directory** to `deployment_version` in the project settings. This tells Railway to treat `deployment_version/` as the project root so it finds `Procfile` and `requirements.txt` directly.
3. In the Railway project's **Variables** tab, add the following environment variable:

   | Variable | Required | Value |
   |---|---|---|
   | `GROQ_API_KEY` | Yes | Your Groq API key |
   | `GPT_API_KEY` | No | OpenAI key (users can also enter it in the UI) |
   | `GROQ_MODEL` | No | Defaults to `llama-3.1-8b-instant` |

4. Railway reads `Procfile` and starts the server automatically:
   ```
   web: python backend/server.py
   ```
5. Once the deploy completes, go to **Settings → Networking → Generate Domain** to get your public HTTPS URL (e.g. `https://pact-production-xxxx.up.railway.app`).

**Verifying the backend:**

Visit `https://your-railway-url.up.railway.app/status`. You should see:
```json
{
  "loaded": true,
  "probe_ready": true,
  "backend": "groq",
  "groq_api_key_set": true
}
```

The startup logs should include:
```
AU probe: MiniLM probe loaded (w=torch.Size([384]), b=-3.8187).
Groq backend ready: model=llama-3.1-8b-instant
```

If `probe_ready` is false, confirm that `data/au_probe/minilm_probe.pt` is committed to the repository.

---

### Part 2 — Frontend on GitHub Pages

The frontend is a static site. No server is needed.

1. Open `deployment_version/frontend/config.js` and set `BACKEND_URL` to your Railway HTTPS URL:
   ```js
   const BACKEND_URL = "https://your-railway-url.up.railway.app";
   ```
2. Commit and push the change to GitHub.
3. In your GitHub repository settings, go to **Pages** and set the source branch. If the frontend files (`index.html`, `app.js`, `style.css`, `config.js`) are in the repo root or a specific folder, point Pages at that location.
4. GitHub Pages will serve the frontend at `https://yourusername.github.io/your-repo/`.

The frontend URL and the Railway backend URL are the two pieces that must match. `config.js` is the only file that needs updating when the Railway URL changes.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes | Groq API key for Llama inference |
| `GPT_API_KEY` | No | OpenAI API key for GPT-4o-mini. Users can also provide this in the UI. |
| `GROQ_MODEL` | No | Groq model to use. Defaults to `llama-3.1-8b-instant`. |
| `PORT` | No | Injected automatically by Railway. Do not set manually. |

---

## Authors

Anushree Udhayakumar, Gawon Lim, Jesse Marsh

IS597 — Human-Centered Data Science, University of Illinois Urbana-Champaign
