# PROJECT.md — What Is This Project?

## Overview

This project is a **RAG (Retrieval-Augmented Generation) System** — a full-stack
web application that lets users ask questions and get AI-generated answers
**grounded in their own uploaded documents**.

Instead of relying purely on an LLM's internal knowledge (which can hallucinate),
the system:

1. **Ingests** documents (PDF, DOCX, XLSX, TXT, MD, CSV) by splitting them into
   chunks and converting each chunk into a text **embedding** (a numeric vector).
2. **Stores** those vectors in a PostgreSQL database with the **pgvector**
   extension (hosted on **Neon**).
3. On a question, **retrieves** the most relevant chunks via **cosine
   similarity** vector search.
4. **Generates** an answer with an LLM using *only* the retrieved chunks as
   context, with inline citations like `[Source 1]`.
5. Returns the answer **plus the list of source chunks** (with similarity
   scores), so users can verify where each statement came from.

The result: answers that are traceable, citation-backed, and far less likely to
be invented by the model.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | **React 19** + **TypeScript** + **Vite**, Tailwind CSS 4, React Router 7, lucide-react icons |
| Backend | **Python / FastAPI** (0.115) served with **Uvicorn**, SQLAlchemy 2.0 |
| Database | **Neon PostgreSQL** with **pgvector** (vector column dimension: 768) |
| Embeddings | **Gemini Embedding 2** (`gemini-embedding-2`, 768 dims) via `GEMINI_API_KEY` |
| Primary LLM | **Gemini** (`gemini-3.8-flash`) for answer generation |
| Fallback LLM | **NVIDIA Nemotron 3 Ultra** (`nvidia/nemotron-3-ultra-550b-a55b`) routed through **OpenRouter** (OpenAI-compatible API) |
| Auth | JWT (python-jose, passlib/bcrypt), HS256, 30-minute access tokens |
| Document parsing | pypdf (PDF), python-docx (Word), openpyxl (Excel) |
| Migrations | Alembic |

---

## Repository Structure

```
rag-system/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI entry point, CORS, routers, lifespan
│   │   ├── core/
│   │   │   ├── config.py        # All env-var settings (Settings class)
│   │   │   └── database.py      # SQLAlchemy engine/session
│   │   ├── models/
│   │   │   └── chunk.py         # DocumentChunk model (pgvector column)
│   │   ├── api/
│   │   │   ├── auth.py          # Signup / login -> JWT tokens
│   │   │   ├── documents.py     # Document upload & listing (ingestion)
│   │   │   ├── notebooks.py     # Notebook management
│   │   │   ├── retrieval.py     # POST /api/retrieve (vector search only)
│   │   │   ├── chat.py          # POST /api/chat (end-to-end RAG)
│   │   │   ├── chat_ws.py       # WebSocket /api/chat/ws (streaming RAG)
│   │   │   └── chat_history.py  # Saved chat / answer history
│   │   ├── services/
│   │   │   ├── embedding.py     # Gemini Embedding 2 (768-dim vectors)
│   │   │   ├── retrieval.py     # Top-K pgvector cosine-similarity search
│   │   │   ├── generation.py    # Gemini generation + Nemotron fallback
│   │   │   └── rag.py           # RAGService: retrieval -> context -> answer
│   │   ├── schemas/             # Pydantic request/response models
│   │   └── utils/seed.py        # Sample data seeding
│   ├── alembic/                 # DB migrations
│   ├── requirements.txt
│   ├── .env.example
│   └── test_fallback.py         # Mocked tests for the LLM fallback logic
├── frontend/
│   └── src/
│       ├── pages/               # Landing, Login, Signup, Dashboard, Chat,
│       │                        # Documents, Notebooks, NotebookDetail, Notes,
│       │                        # SavedAnswers, Insights, Settings, Profile, Workspace
│       ├── components/          # Shared UI (ErrorBoundary, Modal, SearchModal, ...)
│       ├── context/             # React context (auth state, etc.)
│       └── services/api.ts      # Typed API client for the backend
└── README.md
```

---

## The RAG Pipeline (Step by Step)

```
User question
     │
     ▼
[1] Embed the question  ──►  Gemini Embedding 2 (768-dim vector)
     │
     ▼
[2] Vector search  ──►  pgvector cosine similarity in PostgreSQL (Neon)
     │                   returns Top-K chunks above the similarity threshold
     ▼
[3] Threshold check
     ├── no chunks pass  ──►  return "I could not find the answer..."
     │                       (LLM is NEVER called — no hallucinations)
     ▼
[4] Build context string  ──►  labelled "[Source N]" blocks, capped at
     │                         RAG_MAX_CONTEXT_CHARS (12000 by default)
     ▼
[5] Generate answer  ──►  Gemini (primary) with a strict system prompt:
     │                    "answer ONLY from the context, cite [Source N]"
     │
     ├── Gemini call fails (e.g. 503)  ──►  Nemotron fallback via OpenRouter
     ▼
[6] Response  ──►  { answer, sources[], found_context, model }
```

### Key design rules

- **No-context, no-LLM**: if no chunk clears `RAG_MIN_SIMILARITY` (default 0.3),
  the LLM is never called and a fixed safe message is returned.
- **Grounding**: the system prompt forbids outside knowledge and requires inline
  citations (`[Source 1]`).
- **Fallback is generation-only**: Nemotron is used **only** for answer
  generation when Gemini fails *before producing output*. Embeddings, retrieval,
  ingestion and auth are never affected. `LLM_FALLBACK_ENABLED` can disable it.
- **Streaming safety**: the primary stream's first tokens are buffered
  (`STREAM_FLUSH_THRESHOLD_CHARS`, 40 chars) before anything is sent over the
  WebSocket. If Gemini fails while the buffer is still holding, the partial
  output is discarded and the fallback streams a clean, single-model answer.
  If Gemini fails *mid-stream* (after tokens were already sent), the fallback
  is **not** started, to avoid stitching two different model outputs together.
- **Reasoning hygiene**: Nemotron is a reasoning model — its budget is padded 4×
  (`FALLBACK_MAX_TOKENS_MULTIPLIER`) and any `ónde…ónde` reasoning blocks /
  `reasoning_content` fields are stripped; users only see final answers.
- **Error sanitization**: raw provider errors (API keys, internal details) are
  logged server-side only; users get a generic "service temporarily unavailable"
  message.

---

## API Endpoints (FastAPI)

All endpoints (except health check) require a **JWT bearer token** from
`POST /api/auth/login`.

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/auth/*` | Signup / login, returns JWT |
| POST | `/api/documents/upload` | Upload PDF/DOCX/XLSX/TXT/MD/CSV → chunk + embed + store |
| GET | `/api/documents` | List uploaded documents |
| POST | `/api/retrieve` | Vector search only — return Top-K relevant chunks |
| POST | `/api/chat` | End-to-end RAG — question in, grounded answer + sources out |
| WS | `/api/chat/ws` | Streaming RAG — `start` → `context` → `answer_chunk`… → `complete` |
| GET | `/api/chat_history/*` | Saved answers / chat history |
| GET | `/api/notebooks/*` | Notebook management |
| GET | `/` and `/health` | Health checks |

Interactive Swagger UI: **http://127.0.0.1:8000/docs**

### Example: `POST /api/chat`

Request:
```json
{
  "question": "What is the capital of France?",
  "top_k": 5,
  "min_similarity": 0.3
}
```

Response:
```json
{
  "question": "What is the capital of France?",
  "answer": "The capital of France is Paris [Source 1].",
  "sources": [
    {
      "chunk_id": 12,
      "document_id": "doc_france",
      "chunk_index": 1,
      "similarity_score": 0.87,
      "snippet": "The capital of France is Paris, known for the Eiffel Tower..."
    }
  ],
  "found_context": true,
  "model": "gemini-3.8-flash"
}
```

`model` reports the model that **actually** produced the answer (Gemini, or the
Nemotron fallback if it fired).

---

## Frontend (React)

The frontend is a modern React 19 SPA (Vite + Tailwind CSS 4) providing a
document-chat workspace:

- **Landing / Login / Signup** — JWT-based auth screens
- **Dashboard / Workspace** — main app shell (Sidebar, Topbar, SearchModal)
- **Documents** — upload & manage the knowledge base
- **Notebooks / NotebookDetail** — organize documents by notebook
- **Chat** — ask questions, see token-by-token streaming answers (WebSocket)
  with expandable source citations
- **SavedAnswers / Notes / Insights** — review past answers and takeaways
- **Settings / Profile** — account & app settings
- **ErrorBoundary** — graceful error recovery

Configured to talk to the backend at `http://0.0.0.0:8000` (CORS allows the
Vite dev server on `localhost:5173`).

---

## Configuration (backend/.env)

Key settings (see `backend/.env.example` for the full template):

```env
DATABASE_URL=postgresql://...:...@.../...?sslmode=require   # Neon Postgres
GEMINI_API_KEY=...            # embeddings + primary LLM
GEMINI_LLM_MODEL=gemini-3.8-flash
OPENROUTER_API_KEY=...        # fallback LLM (Nemotron)
LLM_FALLBACK_ENABLED=true
RAG_TOP_K=5
RAG_MIN_SIMILARITY=0.3
RAG_MAX_CONTEXT_CHARS=12000
JWT_SECRET_KEY=...            # change in production
CHUNK_SIZE=1000
CHUNK_OVERLAP=150
```

Notes:
- ⚠️ The pgvector column is **fixed at 768 dimensions** — matching
  `gemini-embedding-2`. Never change `EMBEDDING_DIMENSION` without a migration.
- `config.py` provides compatibility aliases (`GOOGLE_*`) so the document
  ingestion modules share the same Gemini key/model as the retrieval pipeline.

---

## Running the Project

### Backend

```bash
cp backend/.env.example backend/.env      # then fill in real values
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --reload     # Swagger: http://127.0.0.1:8000/docs
```

On startup the app creates tables and (in `simple-hash` dev mode) seeds sample
data.

### Frontend

```bash
cd frontend
npm install
npm run dev                               # http://localhost:5173
```

### Tests

```bash
cd backend
python test_fallback.py                   # mocked fallback tests (no live API calls)
```

The test suite (9 scenarios) verifies the fallback rules: Gemini-success skips
fallback, Gemini-failure uses Nemotron with the same prompt, both-fail gives a
sanitized error, streaming mid-fail behavior, think-block stripping, disabled
fallback, and no-leak guarantees.

---

## In One Sentence

> A full-stack, JWT-secured RAG platform: upload documents → they're chunked and
> embedded into Postgres/pgvector → ask questions → get streamed, citation-backed
> answers from Gemini, with an automatic Nemotron fallback when Gemini fails.
