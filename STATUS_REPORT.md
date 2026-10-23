# Status Report: WebSocket Event Loop Blocking Fix

## Overview
Fixed the WebSocket event-loop blocking issue in the RAG system's streaming chat path. The core question-answering pipeline now runs blocking operations on worker threads instead of the event loop.

## Changes Made

### 1. `backend/app/services/ws_rag.py` (Retrieval in worker thread)
- **Problem**: `RetrievalService.retrieve()` (Gemini embedding HTTP call + pgvector DB query) was called directly on the event loop, freezing all connected WS streams while retrieval was in flight.
- **Fix**: Detection of sync vs async `retrieve()` method, with synchronous calls dispatched via `asyncio.to_thread()`. Async (coroutine) calls are awaited directly on the loop, preserving existing test behavior.
- **Verified**: All generation-fallback tests pass (7/7).

### 2. `backend/app/api/chat_ws.py` (DB queries in worker thread)
- **Problem**: Four remote-DB round-trips (user lookup, notebook check, session ownership, history persistence) blocked the event loop simultaneously with retrieval.
- **Fix**: All synchronous `db_session.query()` calls wrapped in `await asyncio.to_thread(lambda: ...)`.
- **Result**: Individual users' remote Postgres queries no longer stall other connected clients.

### 3. `backend/app/services/generation.py` (Non-blocking Gemini stream)
- **Problem**: `_stream_gemini()` used the synchronous google-genai SDK iterator directly, each `next()` blocking the event loop.
- **Fix**: 
  - Stream opened and first chunk pulled via `asyncio.to_thread()`.
  - Remaining chunks pumped by a daemon thread into a `queue.Queue`, consumed asynchronously via `asyncio.to_thread(q.get)`.
  - Same pattern as the already-working `_stream_nemotron` fallback.
- **Verified**: 7 streaming-fallback tests pass; 34/34 isolation/documents/security tests pass; 32/32 fallback assertions pass.

### 4. `backend/tests/test_ws_chat.py` (Syntax error + test mismatches)
- **Problem**: Pre-existing syntax error (missing newline before `def`) prevented the entire WS test file from being collected. Other test assertions used `sources[0]["content"]` but the real API always returned `snippet`.
- **Fix**: 
  - One-line syntax fix enabling test collection.
  - Assertion changes: `content` → `snippet` in 4 test checks.
  - Fixture finalizer changed from `patcher.__exit__()` → `patcher.stop()` (Python 3.14 compatibility: pytest calls finalizers with no args; `_patch.__exit__` forwards `()` to `ExitStack.__exit__()` which needs exception details).
  - Added `stub_gen.set_answer()` for per-test streamed answer control.
  - All 13 WS flow tests now pass.

## Verification Results

| Test Suite | Result |
|---|---|
| `tests/test_stream_fallback.py` | 7/7 pass ✓ |
| `tests/test_chat_isolation.py` + `test_documents_api.py` + `test_security_isolation.py` | 34/34 pass ✓ |
| `test_fallback.py` | 32/32 assertions pass ✓ |
| `tests/test_ws_chat.py` | 13/13 pass ✓ |
| Generation Gemini streaming (reworked) | Correctly non-blocking ✓ |

## Symptom Impact

| Symptom | Before Fix | After Fix |
|---|---|---|
| Slows with multiple WS users | Random multi-second freezes | Other users stay responsive |
| Very slow answers (10-30s) | Gemini retry backoff could add 8s+ before any generation | Same backoff, but other WS streams never blocked |
| Gets slower as documents accumulate | Sequentially scanned all chunks | Same, but retrieval path no longer freezes loop |

## What's NOT changed (pre-existing / out of scope)

- **Retry backoff**: `GEMINI_LLM_MAX_RETRIES=3` with 1s/2s/4s delays + jitter remains. If Gemini is degraded, the worst-case wait before fallback is still ~8s of sleep across retries. This is the #1 contributor to "sometimes very slow" answers and requires a separate config change if desired.
- **Answer verbosity**: The `SYSTEM_INSTRUCTION` rule to "be thorough and complete" still applies; shorter answers would require a prompt change.
- **Missing pgvector ANN index**: Sequential scan on `document_chunks.embedding` for every question. With many chunks, adding an HNSW index would help throughput but isn't the event-loop blocking issue.

## Files Modified

- `backend/app/services/ws_rag.py` — retrieval in worker thread
- `backend/app/api/chat_ws.py` — DB queries in worker thread
- `backend/app/services/generation.py` — non-blocking Gemini stream
- `backend/tests/test_ws_chat.py` — syntax fix, assertion updates, fixture finalizer

## Next Steps (if desired)

1. Reduce Gemini retry delays (`GEMINI_LLM_RETRY_BASE_DELAY`) for faster fallback.
2. Add `HNSW` index on `document_chunks.embedding` for faster retrieval scans.
3. Tighten the system prompt in `generation.py` for shorter answers.
4. Add a proper `created_at` timestamp or size estimate to help gauge question complexity.