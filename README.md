# Greenko Group — Energy Operations Document RAG Assistant

A small, reproducible retrieval-augmented document assistant for the supplied renewable-energy operations, maintenance and safety notes.

## Problem and intended user
Operations and maintenance staff often need to locate a specific instruction across short SOPs and operational notes. This prototype accepts a natural-language question, retrieves relevant passages from the bundled documents, and returns an answer with source filenames and visible retrieval evidence. It is a decision-support prototype, not an operational control system.

## Architecture

```text
Markdown source documents
        ↓
Load + remove title markup
        ↓
Word chunking (110 words, 25-word overlap by default)
        ↓
TF-IDF vectorization (unigrams + bigrams)
        ↓
Cosine-similarity retrieval (top-k)
        ↓
Relevance threshold / abstention
        ↓
Optional LLM answer (if API key configured)
        ↓ otherwise
Grounded extractive fallback
        ↓
FastAPI JSON response + source citations + frontend
```

### Method choices
- **Task:** document retrieval and grounded question answering (RAG), not forecasting or classification.
- **Chunking:** word-based chunks with overlap preserve a little context across chunk boundaries. The bundled files are very short, so most become one chunk; the strategy is configurable for larger documents.
- **Embeddings/retrieval baseline:** scikit-learn TF-IDF sparse vectors with unigram/bigram features and cosine similarity. This is lightweight, deterministic, runs locally and needs no model download. It is a lexical vector baseline, not a transformer semantic embedding model; this is a known limitation.
- **Generation:** when `OPENAI_API_KEY` is configured, retrieved passages are sent to an OpenAI-compatible chat-completions endpoint with instructions to answer only from evidence and cite filenames. Without a key, the system returns the retrieved source statements verbatim as a safe, grounded fallback.
- **Hallucination control:** low-relevance questions are abstained from; the LLM receives only retrieved evidence; response includes evidence and source names; operational safety disclaimer is always included. This is a lightweight guardrail, not a formal guarantee that an LLM cannot hallucinate.

## Repository layout

```text
backend/app.py       FastAPI API, ingestion, chunking, retrieval, generation, evaluation
frontend/index.html  Browser UI
 data/*.md           Bundled source documents
 data/questions.csv  Small retrieval evaluation set
Procfile             Railway start command
requirements.txt     Python dependencies
```

## Run locally

Python 3.10+ is recommended.

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn backend.app:app --reload --port 8000
```

Open http://127.0.0.1:8000. API docs: http://127.0.0.1:8000/docs.

## API
- `GET /api/health` — readiness and index details.
- `GET /api/demo` — dataset and implementation summary.
- `POST /api/run` — JSON body, e.g. `{"text":"How should gearbox vibration be investigated?","top_k":3}`.
- `GET /api/evaluate` — checks top-1 retrieved source against the bundled expected-source labels.

## Optional LLM configuration
Set these as environment variables in your local shell or Railway service. Never commit a real key.

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_MODEL="gpt-4o-mini"
# Optional for an OpenAI-compatible provider:
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

If the key is absent or the request fails, the app uses its grounded extractive fallback. The default demo therefore works without external API access.

## Evaluation and validation
Run `GET /api/evaluate` after starting the server. The bundled `questions.csv` has only two labelled questions. Report its measured top-1 source accuracy exactly as returned; do not present this tiny smoke test as a statistically meaningful benchmark. Also test an unrelated question and confirm the assistant abstains. Suggested manual checks:

1. Inverter derating question → `solar_inverter_sop.md`.
2. Gearbox vibration question → `wind_turbine_maintenance.md`.
3. Spillway gate safety question → `hydro_safety_brief.md`.
4. Generation deviation question → `grid_operations_note.md`.
5. An unsupported question (e.g. "What is the capital of France?") → abstention.

## Assumptions and limitations
- The supplied four Markdown notes are the complete minimum knowledge base; no external energy facts have been added.
- TF-IDF is lexical and may miss paraphrases or domain synonyms. A production version should evaluate a domain-appropriate sentence-transformer embedding model and a vector database on a larger labelled dataset.
- The two provided evaluation examples are too few to establish robust retrieval quality. Add expert-labelled questions and relevance judgments.
- LLM output depends on the configured provider/model. Validate citations and claims against evidence before operational use.
- No authentication, role-based access, document versioning, audit logging, rate limiting, or monitoring is implemented.
- Never use this prototype as a substitute for current approved safety procedures or authorized operational judgment.

## Railway deployment
1. Push this repository to GitHub.
2. In Railway, create a new project and deploy from the GitHub repository.
3. Railway detects the included `Procfile`; the start command is `uvicorn backend.app:app --host 0.0.0.0 --port $PORT`.
4. Add `OPENAI_API_KEY` only if you intend to enable provider-based generation. The app also runs without it.
5. Wait for deployment, open the generated public domain, and verify `/api/health`, the UI, a supported question and an unsupported question.
6. Add the deployed URL to your submission notes. Deployment must be completed and verified in your own Railway account; this repository alone is not evidence of a live deployment.

## Submission checklist
- [ ] Push the full repository to GitHub and confirm it is public or share access with the reviewer.
- [ ] Run locally and capture screenshots of a supported answer and an abstained question.
- [ ] Record the actual `/api/evaluate` output.
- [ ] Deploy to Railway and verify the live URL.
- [ ] Remove secrets and local environment files before pushing.

### Observed local smoke-test result (November 12, 2026)
- `GET /api/health`: HTTP 200; 4 documents and 4 chunks indexed.
- `GET /api/evaluate`: 2/2 expected source matches (top-1 source accuracy 1.00) on the supplied two-question set.
- Manual API checks: inverter derating → solar SOP; spillway inspection → hydro safety brief only after relevance filtering; unsupported geography question → abstained with no sources.
- LLM provider was not configured during local validation; these checks exercised the grounded extractive fallback. Results are limited by the tiny bundled dataset.

Run automated API tests with `pip install pytest httpx && pytest -q`.
