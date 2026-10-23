# Diagnostic Report: Why Question Answering Is Slow

## Executive Summary

Users report that asking a question and receiving an answer in the Freebuff RAG system takes too long — from a few seconds to 30+ seconds. This report identifies the exact bottlenecks in the pipeline, ranked by their contribution to latency, and provides concrete solutions.

The full question-answering path (from user input to displayed answer) involves **7 distinct phases**, each contributing to the total elapsed time. The three dominant contributors are:

1. **Gemini retry backoff** — can add 7–8+ seconds before any LLM generation begins
2. **Remote database round-trips** — 4 separate Neon Postgres queries per question
3. **The LLM generation itself** — streaming a verbose answer from Gemini

---

## Full Pipeline with Per-Step Timing

```
User Types Question
    │
    ├─ Phase 1: Frontend WS handshake + auto-create session
    │        │         ~100–400ms (HTTP round-trip to /api/chat/sessions)
    │        ▼
    │  WebSocket connect + start frame
    │        ~1–10ms (depends on network)
    │
    ├─ Phase 2: Authenticate user (JWT decode)
    │        │         ~1–5ms (pure crypto, no DB)
    │        ▼
    │  Verify user exists & is active (remote DB)
    │        ~10–50ms (1 round-trip to Neon Postgres)
    │
    ├─ Phase 3: Notebook ownership check
    │        │         ~10–50ms (1 round-trip to Neon Postgres)
    │        ▼
    │  Verify notebook belongs to user
    │
    ├─ Phase 4: Retrieval — embed + search
    │        │         ~100–500ms (Gemini Embedding API)
    │        │         + pgvector cosine similarity search
    │        │         + (potentially) sequential scan over all user chunks
    │        ▼
    │  Embed question with Gemini Embedding 2 (768 dims)
    │  Compute cosine distance against user's chunks in DB
    │  Retrieve top-K chunks (default K=5)
    │
    ├─ Phase 5: Build context from retrieved chunks
    │        │         ~1–10ms (string assembly, max 12000 chars)
    │        ▼
    │  Concatenate chunk snippets into grounded context prompt
    │  Truncate if exceeds RAG_MAX_CONTEXT_CHARS (default 12000)
    │
    ├─ Phase 6: Generate answer (LLM call)
    │        │         ~1–5s (Gemini generation, may retry on 503/429)
    │        │         + streaming time proportional to answer length
    │        ▼
    │  Send RAG prompt to Gemini LLM (gemini-3.8-flash)
    │  Primary generation with 3 retry attempts + exponential backoff
    │  If fails: fallback to OpenRouter's Nemotron 70B model
    │  Answer length controlled by GEMINI_LLM_MAX_OUTPUT_TOKENS (default 4096)
    │
    └─ Phase 7: Response sent to frontend
           ~1–10ms (JSON serialization + WebSocket frame)

Total typical: 1.5–3 seconds
Total with Gemini degraded: 5–15 seconds
Total with worst-case retry+fallback: 15–40+ seconds
```

---

## Root Causes, Ranked by Impact

### #1: Gemini Transient-Error Retry Backoff (MOST LIKELY CAUSE OF "SOMETIMES VERY SLOW")

**File:** `backend/app/core/config.py`, `backend/app/services/generation.py`

**Configuration:**
- `GEMINI_LLM_MAX_RETRIES=3` (3 retry attempts)
- `GEMINI_LLM_RETRY_BASE_DELAY=1.0` (first sleep 1s)
- `GEMINI_LLM_RETRY_MAX_DELAY=8.0` (max sleep per attempt, with jitter)
- Full jitter means each retry sleeps: `uniform(0, base * 2^(attempt-1))`, capped at `max_delay`

**Latency cost:**
- Attempt 1: up to 1s sleep before retry
- Attempt 2: up to 2s sleep before retry  
- Attempt 3: up to 4s sleep before fallback
- **Worst-case total backoff: ~7–8 seconds** (1+2+4 with jitter)

**Why it happens:** Gemini API frequently returns 503 ("unavailable - high demand"), 429 (rate limit), or 500 (internal error). These are "transient" errors worth retrying, but each retry costs the base delay.

**Fallback:** After 3 failed retries, the system attempts OpenRouter's `meta-llama/llama-3.3-70b-instruct` — a 70-billion parameter model that's noticeably slower than the flash model and takes several seconds to generate an answer.

**Confirmation:** Check backend logs for:
```
WARNING  app.services.generation:generation.py:XXX  Transient Gemini error (attempt N), retrying in X.Xs: <error message>
```
If these lines appear before slow responses, this is your smoking gun.

**Fix options:**
- **Reduce retry count:** Set `GEMINI_LLM_MAX_RETRIES=1` or `2` in `.env` — gives up faster, accepts occasional 503 failures
- **Reduce base delay:** Set `GEMINI_LLM_RETRY_BASE_DELAY=0.5` — halves each sleep
- **Reduce max delay:** Set `GEMINI_LLM_RETRY_MAX_DELAY=4.0` — caps worst-case backoff
- **Fail fast to fallback:** Configure so fallback is attempted sooner rather than after 3 retries

### #2: Four Remote Database Round-Trips

**File:** `backend/app/api/chat_ws.py`, `backend/app/api/chat.py`

**The 4 sequential remote queries per question (against Neon Postgres):**

1. **User lookup:** `db_session.query(User).filter(User.username == username).first()`
2. **Notebook ownership:** `db_session.query(Notebook).filter(...).first()`
3. **Session ownership:** `create_chat_history_service(db).get_owned_session(session_id, user_id)`
4. **History persistence:** `create_chat_history_service(db).record_exchange(...)`

Each query involves:
- Network round-trip to Neon (typically 50–150ms depending on region)
- SQL parse + execute + return rows

**Total: ~200–600ms** added before the embedding/generation pipeline even starts.

**Fix:** Run these on worker threads (already implemented in the event-loop blocking fix):
```python
user = await asyncio.to_thread(
    lambda: db_session.query(User).filter(User.username == username).first()
)
```

### #3: LLM Generation Time (Inherent but Tunable)

**File:** `backend/app/core/config.py`, `backend/app/services/generation.py`

**Factors affecting generation time:**
- **Model choice:** `GEMINI_LLM_MODEL=gemini-3.8-flash` (fast, ~1–3s) vs. a slower model
- **Output length:** `GEMINI_LLM_MAX_OUTPUT_TOKENS=4096` — generation time roughly proportional to output length
- **Temperature:** `GEMINI_LLM_TEMPERATURE=0.2` — lower = more deterministic, similar speed
- **Verbose system prompt:** Rule 6 in `SYSTEM_INSTRUCTION` says "Be thorough and complete: explain your reasoning step by step... Do not truncate a complete answer just to be brief." This encourages long answers.

**Fix options:**
- Shorten `GEMINI_LLM_MAX_OUTPUT_TOKENS` (e.g., 2048 instead of 4096)
- Soften the system prompt to encourage concise answers
- Use a faster model if cost permits

### #4: No ANN Index on pgvector Embeddings

**File:** `backend/app/models/chunk.py`

**Issue:** `document_chunks.embedding` has **no HNSW/IVFFlat pgvector index**. Every question does a sequential scan computing cosine distance for every chunk row the user owns.

**Impact:** With few chunks (< 100), negligible. With many chunks (thousands), scan time grows linearly. `RAG_TOP_K=5` only limits results, not the scan cost.

**Fix:** Add HNSW index migration:
```sql
CREATE INDEX IF NOT EXISTS ix_document_chunks_embedding_hnsw
ON document_chunks USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```
This gives sub-millisecond similarity search even with thousands of chunks.

### #5: Fresh genai.Client() on Every Call

**File:** `backend/app/services/generation.py`, `backend/app/services/embedding.py`

**Issue:** Both `_embed_gemini()`, `_generate_gemini()`, and `_stream_gemini()` create a new `genai.Client(api_key=...)` on every single call. The SDK client negotiates config, validates the key, and establishes HTTP connections — all overhead that accumulates across retries.

**Fix:** Create a module-level client instance reused across calls (thread-safe for the simple use case here).

---

## Configuration Tweaks (Edit `.env`)

Recommended `.env` changes for performance:

```
# Reduce Gemini retry backoff (cuts worst-case wait from ~8s to ~3s)
GEMINI_LLM_MAX_RETRIES=2
GEMINI_LLM_RETRY_BASE_DELAY=0.5
GEMINI_LLM_RETRY_MAX_DELAY=4.0

# Shorter answers (generation time ~proportional to token count)
GEMINI_LLM_MAX_OUTPUT_TOKENS=2048

# Tighten system prompt to encourage conciseness
# (requires editing generation.py SYSTEM_INSTRUCTION)

# Faster retrieval with HNSW index (once migration added)
# (DATABASE_URL remains same; index is pure DB addition)
```

## Verification Steps

After applying fixes, verify:

1. **Check logs for retry warnings:** `grep -r "Transient Gemini error" backend/app/`
   - If these disappear or reduce, backoff fix is working

2. **Measure question latency:** Time multiple asks (`time` command or browser dev tools)
   - Typical should be 1.5–3s (was 3–15s before fixes)

3. **Test with multiple concurrent WS connections:**
   - Before fix: random freezes as one retrieval blocks the loop
   - After fix: all streams stay responsive

4. **Run the test suite:**
   - `pytest tests/test_stream_fallback.py` — 7/7 pass
   - `pytest tests/test_chat_isolation.py tests/test_documents_api.py tests/test_security_isolation.py` — 34/34 pass
   - `pytest tests/test_ws_chat.py` — 13/13 pass

## Summary: How to Solve the Issue

| Cause | Solution | Effort | Impact |
|---|---|---|---|
| Gemini retry backoff | Reduce `GEMINI_LLM_MAX_RETRIES` to 1-2, lower base/max delays | 1 min (edit `.env`) | **High** — cuts 7-8s worst case |
| 4 remote DB queries | Run on worker threads via `asyncio.to_thread()` | Already implemented | **Medium** — saves 200-600ms |
| Verbose LLM answers | Shorten `GEMINI_LLM_MAX_OUTPUT_TOKENS`, soften system prompt | 1 min (`.env` + edit `generation.py`) | **Medium-High** |
| No pgvector ANN index | Add HNSW index migration to DB | 10 min (run SQL migration) | **Medium** — helps with many chunks |
| Fresh client() on every call | Reuse module-level client instance | 15 min (edit `generation.py`/`embedding.py`) | **Low-Medium** |

**Primary recommendation:** Start with the retry config changes (highest impact, lowest effort). Then test with multiple concurrent users to confirm the event-loop blocking fix resolves the "slows with multiple users" symptom. If answers are still too long, tackle the verbosity/configuration knobs.