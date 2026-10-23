# Analysis: Why answering a question is slow

Deep check of the full question pipeline — frontend → `POST /api/chat` / `WS /api/chat/ws` → `WSRAGOrchestrator` → embedding → pgvector → Gemini → fallback → response.

---

## The full path a question takes, with time-cost per step

```
frontend: WS handshake + start frame (~ms)
  → 3–4 sequential DB round-trips (remote Neon Postgres)     ~100–800ms
  → Gemini Embedding API call (sync, uncached)               ~100–500ms
  → pgvector cosine search (no ANN index — seq scan)         small–large (grows with chunks)
  → LLM generation (streaming, but see below)                SECONDS — dominant
```

---

## Findings, ranked by likely impact

### 1. Transient-error retry backoff can add 7–8+ seconds before generation even starts — the most likely "sometimes very slow" cause

**Files:** `backend/app/services/generation.py`, `backend/app/core/config.py`

`GEMINI_LLM_MAX_RETRIES=3`, base delay 1s, cap 8s, full jitter. On a 503/429/timeout (very common on Gemini), it sleeps up to `1+2+4s` (jittered up to 8s each), retries 3 times, and **only then** falls back to OpenRouter's `meta-llama/llama-3.3-70b-instruct` — a 70B model that's noticeably slower than the flash model.

Worst case: ~8s of backoff sleep + 4 slow Gemini attempts + a slow 70B fallback ≈ **20–40s**. If Gemini is consistently degraded, *every* question pays this.

### 2. The WS route blocks the event loop

**File:** `backend/app/services/ws_rag.py` (`WSRAGOrchestrator.handle_question`)

`handle_question()` is `async`, but it calls `self.retrieval_service.retrieve(...)` **directly** — a synchronous method that makes a *synchronous* HTTP call to the Gemini embedding API plus DB queries. It also does the `User`/`Notebook`/session-ownership queries synchronously on the loop.

Consequences:

- While one user's embedding call or DB query is in flight, **every other WebSocket and HTTP request in that event loop stalls**. Under quiet load you won't notice; with 2–3 users you get random multi-second freezes.
- `_stream_gemini` also iterates the **sync** google-genai iterator inside the async generator — each `next()` blocks the loop.
- Contrast: the REST path (`backend/app/api/chat.py`) correctly wraps everything in `asyncio.to_thread`.

### 3. The answer itself is long, and the prompt encourages it

**File:** `backend/app/services/generation.py` (`SYSTEM_INSTRUCTION`)

Rule 6 says *"Be thorough and complete: explain your reasoning step by step… Do not truncate a complete answer just to be brief."* With `GEMINI_LLM_MAX_OUTPUT_TOKENS=4096`, generation time is roughly proportional to output length — a "thorough step-by-step" answer is several times slower to produce than a concise one.

### 4. Per-request overhead that accumulates before the first token

- **Fresh `genai.Client()` created on every call** in `_embed_gemini` and `_generate_gemini` / `_stream_gemini` — re-negotiates config each time.
- **4 sequential remote-DB round-trips** per ask (`user` lookup, `Notebook` check, `get_owned_session`, then the vector search) plus `pool_pre_ping=True` on the engine adds another round-trip per checkout — all against a remote Neon instance.
- **Frontend auto-creates a chat session first** (`POST /api/chat/sessions` in `frontend/src/pages/Workspace.tsx`) before opening the WS — an extra full HTTP round-trip ahead of the question.

### 5. No ANN index on the embeddings

**File:** `backend/app/models/chunk.py`

`document_chunks.embedding` has **no HNSW/IVFFlat pgvector index** — every question does a sequential scan computing cosine distance for every chunk row the user owns (`RAG_TOP_K=5` doesn't help; the scan is the cost). Negligible with a few hundred chunks, linearly worse as notebooks grow.

### 6. Small stuff

- `STREAM_FLUSH_THRESHOLD_CHARS = 40` — trivial.
- `_stream_nemotron` does `asyncio.to_thread(q.get)` **per chunk** — an executor round-trip per token; adds jitter to fallback streams only.
- `RAG_EXECUTOR` in `backend/app/api/chat.py` is created but never used (dead code; the comment claims it's pinned but the code uses the default executor).

---

## Summary diagnosis

| Symptom | Cause |
|---|---|
| Every answer slow | Long verbose answers (#3) + 4 DB round-trips + embedding call (#4) |
| Occasionally **very** slow (10–30s) | Gemini retry backoff then 70B fallback (#1) |
| Slows down with multiple users / feels random | WS event-loop blocking (#2) |
| Gets slower as documents accumulate | Missing pgvector ANN index (#5) |

---

## How to confirm empirically

1. Set `DEBUG_WS_FLOW=1` in `backend/.env` — it logs exactly which WS messages are yielded (and when).
2. Check backend logs for the warning line `"Transient Gemini error (attempt N), retrying in X.Xs"`.
   If those lines appear before slow responses, **finding #1 is the smoking gun**.

---

## Recommended fixes (highest value first)

1. **Run WS retrieval in a worker thread** (`asyncio.to_thread`), same as the REST path — removes event-loop stalling.
2. **Reuse a single `genai.Client`** per process instead of constructing one per call.
3. **Trim retry delays** (e.g. 2 retries × ~0.5/1s, or fail fast to the fallback) — cuts the worst-case wait dramatically.
4. **Add an HNSW index** on `document_chunks.embedding` (`CREATE INDEX ... ON document_chunks USING hnsw (embedding vector_cosine_ops)`).
5. **Tighten the system prompt** (drop or soften rule 6) for shorter, faster answers.
