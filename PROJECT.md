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
| Auth | JWT (python-jose, passlib/bcrypt), HS256, 30-minute access tokens + rotating DB-tracked refresh cookie |
| Document parsing | pypdf (PDF), python-docx (Word), openpyxl (Excel) |
| Migrations | Alembic |

---

## Repository Structure

```
rag-system/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI entry point, CORS, lifespan, routers
│   │   ├── api/
│   │   │   ├── __init__.py          # re-exports all router instances
│   │   │   ├── auth.py              # register / login / me / logout / refresh + cookie helpers
│   │   │   ├── documents.py         # upload + metadata listing + legacy listing
│   │   │   ├── notebooks.py         # notebooks, documents, notes, saved answers, stats
│   │   │   ├── retrieval.py         # POST /api/retrieve (mandatory notebook scoping)
│   │   │   ├── chat.py              # POST /api/chat (end-to-end RAG + optional history)
│   │   │   ├── chat_ws.py           # WebSocket /api/chat/ws (streaming RAG)
│   │   │   └── chat_history.py      # chat session CRUD
│   │   ├── core/
│   │   │   ├── config.py            # Settings: all env-backed config + GOOGLE_* aliases
│   │   │   └── database.py          # SQLAlchemy engine + sessionmaker + get_db dependency
│   │   ├── models/
│   │   │   ├── chunk.py             # DocumentChunk (pgvector column, user_id/notebook_id)
│   │   │   ├── user.py              # User account
│   │   │   ├── notebook.py          # Notebook, DocumentMeta, Note, SavedAnswer
│   │   │   └── refresh_token.py     # server-tracked refresh tokens
│   │   ├── schemas/
│   │   │   ├── auth.py              # register/login/token/user shapes
│   │   │   ├── chat.py              # ChatRequest/ChatSource/ChatResponse
│   │   │   ├── chat_history.py      # session/message/output schemas
│   │   │   └── ws.py                # WebSocket message schemas
│   │   ├── services/
│   │   │   ├── __init__.py          # canonical + Littu ingestion stack exports
│   │   │   ├── embedding.py         # EmbeddingService (gemini / openrouter / simple-hash)
│   │   │   ├── retrieval.py         # RetrievalService (pgvector cosine search, tenant filtering)
│   │   │   ├── generation.py        # GenerationService (Gemini + Nemotron fallback, streaming)
│   │   │   ├── rag.py               # RAGService (retrieval -> context -> answer)
│   │   │   ├── ws_rag.py            # WSRAGOrchestrator (streaming RAG over WebSocket)
│   │   │   ├── chat_history.py      # session/message persistence with ownership gates
│   │   │   ├── document_processor.py # ingestion orchestration (Littu stack)
│   │   │   └── google_embeddings.py # Google embedding service used by ingestion path
│   │   └── utils/
│   │       └── seed.py              # sample data seeding for offline/dev testing
│   ├── alembic/                     # DB migrations
│   ├── tests/
│   │   ├── conftest.py              # throwaway test schema + TestClient + user fixtures
│   │   ├── test_chat_isolation.py   # per-notebook RAG isolation tests
│   │   ├── test_documents_api.py    # documents metadata/legacy/delete/stats tests
│   │   ├── test_security_isolation.py
│   │   └── test_notebook_isolation.py
│   ├── requirements.txt
│   ├── .env.example
│   └── test_fallback.py             # mocked generation/fallback tests (no DB, no live APIs)
├── frontend/
│   ├── package.json
│   ├── vite.config.ts
│   ├── src/
│   │   ├── main.tsx
│   │   ├── App.tsx
│   │   ├── context/
│   │   │   └── AuthContext.tsx      # auth state + refresh-aware session restore
│   │   ├── services/
│   │   │   └── api.ts              # typed API client with 401 refresh-before-logout
│   │   ├── components/
│   │   ├── pages/
│   │   └── utils/
│   └── .env.example
├── README.md
└── PROJECT.md
```

---

## The RAG Pipeline (Step by Step)

```text
User question
     │
     ▼
[1] Embed the question  ──►  Gemini Embedding 2 (768-dim vector)
     │
     ▼
[2] Vector search  ──►  pgvector cosine similarity in PostgreSQL (Neon)
                        restricted to the caller's user_id (+ notebook_id when given)
     │                   returns Top-K chunks above the similarity threshold
     ▼
[3] Threshold check
     ├── no chunks pass  ──►  return "The answer was not found in the documents
     │                       in this notebook." (LLM is NEVER called — no hallucinations)
     ▼
[4] Build context string  ──►  labelled "[Source N]" blocks, capped at
                                RAG_MAX_CONTEXT_CHARS (12000 by default)
     ▼
[5] Generate answer  ──►  Gemini (primary) with a strict system prompt:
                           "answer ONLY from the context, cite [Source N]"
                           retry transient errors (503/429/5xx) with backoff
                           │
                           ├── Gemini call fails before output  ──►  Nemotron fallback via OpenRouter
                           │                                       (same prompt, same context)
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
  (`FALLBACK_MAX_TOKENS_MULTIPLIER`) and any `<think>...</think>` reasoning
  blocks / `reasoning_content` fields are stripped; users only see final answers.
- **Error sanitization**: raw provider errors (API keys, internal details) are
  logged server-side only; users get a generic "service temporarily unavailable"
  message.

---

## API Endpoints (FastAPI)

All endpoints (except health check) require a **JWT bearer token** from
`POST /api/auth/login`.

### Authentication

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/auth/register` | Create an account |
| POST | `/api/auth/login` | Returns JWT + sets refresh cookie |
| GET | `/api/auth/me` | Current user profile |
| POST | `/api/auth/logout` | Revoke refresh token + clear cookie |
| POST | `/api/auth/refresh` | Exchange refresh cookie for new access token (rotation) |

`POST /api/auth/login` accepts both JSON and `application/x-www-form-urlencoded`
bodies, so Swagger UI's OAuth2 password-flow dialog works directly.

### Documents (ingestion)

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/documents/upload` | Upload PDF/DOCX/XLSX/TXT/MD/CSV → validate → chunk → embed → store |
| GET | `/api/documents` | List user's documents (full metadata) |
| GET | `/api/documents/legacy` | Legacy chunk-count-per-document listing |
| GET | `/api/documents/{id}` | One document's metadata |
| DELETE | `/api/documents/{id}` | Delete document metadata + its chunks |

Upload validation includes extension, MIME, hard size cap while reading, and
file-signature checks (magic bytes for PDF/Word/Excel, decodable-text checks for
TXT/MD/CSV). Files are written to a private server-named temp directory and the
metadata + chunks are committed in one transaction.

### Notebooks / notes / saved answers / stats

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/notebooks` | List user's notebooks with counts |
| POST | `/api/notebooks` | Create notebook |
| GET | `/api/notebooks/{id}` | Notebook with documents/notes/saved answers |
| PATCH | `/api/notebooks/{id}` | Rename/recolor/pin |
| DELETE | `/api/notebooks/{id}` | Delete notebook + metadata + chunks |
| GET | `/api/notes` | List notes |
| POST | `/api/notes` | Create note (optional owned notebook link) |
| PATCH | `/api/notes/{id}` | Update note |
| DELETE | `/api/notes/{id}` | Delete note |
| GET | `/api/saved-answers` | List saved AI answers |
| POST | `/api/saved-answers` | Save an answer from chat |
| DELETE | `/api/saved-answers/{id}` | Delete saved answer |
| GET | `/api/stats` | Dashboard stats scoped to the caller |

Notebook references in notes/saved answers are ownership-checked: another user's
notebook id is rejected as `404`, never silently attached.

### Retrieval

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/retrieve` | Vector search only — return Top-K relevant chunks (notebook-scoped) |
| GET | `/api/retrieve/sample` | Sample question for testing |

Retrieval requires `notebook_id` and verifies ownership. Vector search is
restricted in SQL to `user_id` and optionally `notebook_id`. Legacy chunks with
NULL owner are excluded.

### Chat (end-to-end RAG)

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/chat` | Question in → grounded answer + sources out (+ optional session history) |

Request body:

```json
{
  "question": "What is the capital of France?",
  "notebook_id": 1,
  "top_k": 5,
  "min_similarity": 0.3,
  "session_id": 123
}
```

`notebook_id` is mandatory and ownership-checked. `session_id` is optional and
only used when the caller owns that session; otherwise it is ignored and logged.

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

### Streaming Chat (WebSocket)

| Method | Endpoint | Purpose |
|---|---|---|
| WS | `/api/chat/ws` | Streaming RAG — `start` → `context` → `answer_chunk`… → `complete` |

Client→server `start` message:

```json
{
  "type": "start",
  "token": "<jwt>",
  "question": "...",
  "notebook_id": 1,
  "top_k": 5,
  "min_similarity": 0.3,
  "session_id": 123
}
```

Server→client message types:
- `context` — sources + `found_context`
- `answer_chunk` — next answer fragment
- `complete` — streaming finished
- `error` — something went wrong

Chat history persistence for WebSocket happens **after** the `complete` message,
and only when the session is owned by the authenticated user.

Interactive Swagger UI: **http://127.0.0.1:8000/docs**

---

## Authentication and Session Model

### Access token

- Short-lived JWT (default 30 minutes), HS256, `sub = username`.
- Passed as `Authorization: Bearer ...`.
- `GET /api/auth/me` validates it and returns the user profile.

### Refresh token

- Opaque random value issued on login.
- Stored **hashed** in the `refresh_tokens` table, never the raw value.
- Delivered to the browser as an **HttpOnly** cookie (`rag-refresh` by default).
- JavaScript can never read it.
- Cookie attrs (`Secure`, `SameSite`, `Domain`) are environment-configured and
  default to dev-friendly values on plain HTTP, stricter values over HTTPS.

### Rotation / revocation

- `/api/auth/refresh` implements rotation: the presented refresh token is revoked
  immediately and a new one is issued in its place.
- `/api/auth/logout` revokes the current refresh token and clears the cookie.
- The service layer also supports revoking all of a user's active refresh tokens.

### Frontend refresh behavior

`frontend/src/services/api.ts` implements a **single-flight refresh coordinator**:

- On a 401 from a request that carried an Authorization header, it first tries
  `POST /api/auth/refresh` once.
- Concurrent 401s share the same in-flight refresh promise.
- If refresh succeeds, the original request is retried once with the new token.
- If refresh fails, the token is cleared and the user is logged out to avoid
  infinite retry loops.
- Uploads and WebSocket-style flows that do not carry an Authorization header
  rely on the refresh cookie only.

`AuthContext` restores session on mount by calling `GET /api/auth/me` when a
stored token exists, and clears invalid/expired tokens silently.

---

## Notebook and Data Isolation

The current backend is organized around **notebooks as the unit of isolation**:

- Every notebook belongs to a user.
- Document metadata belongs to a notebook.
- Chunks are linked to a document and carry `user_id` + `notebook_id`.
- Uploads are scoped to an owned notebook (or the user's default notebook for
  legacy-friendly behavior).
- Retrieval and chat both require a notebook id and verify ownership.
- Notes and saved answers may reference an optional owned notebook.
- Stats, document lists, note lists, saved-answer lists, and chat sessions are
  all scoped to the authenticated user.

Chunks with NULL owner are treated as unattributed: retrieval ignores them and
the startup migration backfills ownership from documents/notebooks where possible.

---

## Frontend (React)

The frontend is a modern React 19 SPA (Vite + Tailwind CSS 4):

- **Auth**: `AuthContext` + `api.ts` handle login, register, session restore,
  logout, and refresh-aware API calls.
- **Workspace shell**: sidebar, topbar, search, empty states, toasts, common UI.
- **Documents**: upload and manage the knowledge base.
- **Notebooks / NotebookDetail**: organize documents, notes, saved answers.
- **Chat**: ask questions and read streaming answers with source citations.
- **SavedAnswers / Notes / Insights**: review past answers and takeaways.
- **Settings / Profile**: account and app settings.

It is configured to talk to the backend at `http://localhost:8000` (Vite dev
server on `localhost:5173` is allowed by CORS).

---

## Configuration (backend/.env)

Key settings (see `backend/.env.example` for the full template):

```env
DATABASE_URL=postgresql://...:...@.../...?sslmode=require   # Neon Postgres
GEMINI_API_KEY=...            # embeddings + primary LLM
GEMINI_LLM_MODEL=gemini-3.8-flash
GEMINI_LLM_TEMPERATURE=0.2
GEMINI_LLM_MAX_OUTPUT_TOKENS=1024

# Retry transient Gemini generation errors before fallback
GEMINI_LLM_MAX_RETRIES=3
GEMINI_LLM_RETRY_BASE_DELAY=1.0
GEMINI_LLM_RETRY_MAX_DELAY=8.0

OPENROUTER_API_KEY=...        # fallback LLM (Nemotron)
OPENROUTER_LLM_MODEL=nvidia/nemotron-3-ultra-550b-a55b
OPENROUTER_LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_FALLBACK_ENABLED=true

RAG_TOP_K=5
RAG_MIN_SIMILARITY=0.3
RAG_MAX_CONTEXT_CHARS=12000

JWT_SECRET_KEY=...            # change in production
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

REFRESH_TOKEN_EXPIRE_DAYS=7
REFRESH_COOKIE_NAME=rag-refresh
REFRESH_COOKIE_SECURE=false
REFRESH_COOKIE_SAME_SITE=Lax
REFRESH_COOKIE_DOMAIN=

CHUNK_SIZE=1000
CHUNK_OVERLAP=150

MAX_UPLOAD_SIZE_MB=20
MAX_PAGES_PER_DOC=500
MAX_CHUNKS_PER_DOC=2000
MAX_FILENAME_CHARS=255

CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173

APP_HOST=0.0.0.0
APP_PORT=8000
DEBUG=true
```

Notes:
- ⚠️ The pgvector column is **fixed at 768 dimensions** — matching
  `gemini-embedding-2`. Do not change `EMBEDDING_DIMENSION` without a migration.
- `config.py` provides compatibility aliases (`GOOGLE_API_KEY`, `GOOGLE_*`) so
  the document ingestion modules share the same Gemini key/model as the retrieval
  pipeline.
- `CORS_ORIGINS` is read **before** refresh-cookie defaults that depend on the
  dev-vs-production heuristic.
- Backend startup warns if `JWT_SECRET_KEY` is still the placeholder.

---

## Running the Project

### Backend

```bash
cp backend/.env.example backend/.env      # then fill in real values
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --reload     # Swagger: http://127.0.0.1:8000/docs
```

On startup the app creates tables and (in `simple-hash` dev mode) seeds sample
data. It also runs a best-effort runtime migration for chunk `user_id` /
`notebook_id` columns on existing deployments.

### Frontend

```bash
cd frontend
npm install
npm run dev                               # http://localhost:5173
```

### Tests

```bash
cd backend
python test_fallback.py                   # mocked fallback tests (no live API calls, no DB)
```

The test suite verifies the fallback rules: Gemini-success skips fallback,
Gemini-failure uses Nemotron with the same prompt, both-fail gives a sanitized
error, streaming pre-flush vs mid-stream behavior, think-block stripping,
disabled fallback, and no-leak guarantees.

DB-backed tests live in `backend/tests/` and are designed to run against a
throwaway per-session Postgres schema via `conftest.py`. They are currently not
executable in this environment because the test database is not reachable from
here.

---

## Current Status

### Verified working

- Backend imports cleanly.
- `GET /health` returns `{"status": "healthy"}` through FastAPI's `TestClient`.
- `python test_fallback.py` passes: 32 assertions, 0 failures.
- Frontend production build passes: `npm run build`.

### In-progress / not yet verified

- The DB-backed pytest suite (`tests/` collection + endpoint isolation tests)
  could not be run here because the live Postgres test database is not
  reachable in this environment.
- The refresh-token router exists in `app/api/auth.py` as an optional extension
  point and is wired into `main.py`, but the broader refresh-user-experience
  paths in the frontend may still be maturing depending on which pages consume
  them.

---

## Known Limitations

- pgvector dimension is locked to 768.
- Nemotron fallback is generation-only and depends on an OpenRouter key being
  configured when Gemini is unavailable.
- Streaming fallback only covers failures **before** tokens are flushed to the
  client; mid-stream failures keep the existing error behavior.
- DB-backed integration tests cannot be verified without a reachable test
  database.
- The refresh cookie flow is configured and implemented server-side, but full
  end-to-end refresh UX depends on the frontend pages that consume it.
