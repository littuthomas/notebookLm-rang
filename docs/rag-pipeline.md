# RAG Pipeline: Ingestion, Retrieval, and Notebook Scoping

This document explains how document **ingestion** works, how **retrieval** answers
a question, and why (before the fix) a question asked *inside a notebook* could
pull in sources from **other notebooks** as well.

Code lives in:

| Stage          | File(s)                                             |
|----------------|-----------------------------------------------------|
| Ingestion      | `backend/app/api/documents.py`, `backend/app/services/document_processor.py`, `backend/app/services/chunker.py`, `backend/app/services/google_embeddings.py` |
| Retrieval      | `backend/app/services/retrieval.py`, `backend/app/services/rag.py` |
| Generation     | `backend/app/services/generation.py`                |
| Endpoints      | `backend/app/api/chat.py`, `backend/app/api/chat_ws.py`, `backend/app/api/retrieval.py` |
| Frontend calls | `frontend/src/services/api.ts`                      |

---

## 1. How ingestion works

Every uploaded file goes through this pipeline (one transaction — it all
commits or rolls back together):

```
Upload (multipart form: file + notebook_id)
   │
   ├─ 1. Auth & scoping          JWT → user; notebook resolved
   │     - notebook_id form field must belong to the caller
   │     - if missing/invalid → user's oldest notebook (auto-created if none)
   │
   ├─ 2. Text extraction         file_parser.extract_text()
   │     - supports PDF, Word, Excel, TXT, MD, CSV
   │
   ├─ 3. Cleaning                text_cleaner.clean_batch()
   │
   ├─ 4. Chunking                chunker.chunk_pages()
   │     - CHUNK_SIZE=1000 chars, overlap 150 by default
   │     - each chunk keeps { document_id, chunk_index, content }
   │
   ├─ 5. Embedding               Gemini Embedding 2 (768 dimensions)
   │     - chunks that fail embedding are skipped (logged)
   │
   └─ 6. Persistence (documents.py) — the isolation step
         DocumentMeta   → notebook_id, filename, status, page/chunk counts
         DocumentChunk  → content, embedding (Vector 768),
                          user_id  = owner of the document,
                          notebook_id = the notebook it was uploaded to
```

**The critical piece:** each row in `document_chunks` carries **`user_id`**
(who owns it) and **`notebook_id`** (which notebook it belongs to).
These two columns are what make notebook-scoped retrieval possible.

Safety rails on ingest: upload size (20 MB), page (500) and chunk (2000)
limits, private temp dir cleaned in a `finally`.

---

## 2. How retrieval works

When you ask a question (`POST /api/chat`, `/api/chat/ws`, or
`POST /api/retrieval`):

```
Question  +  notebook_id (optional)
   │
   ├─ 1. Auth                     JWT → user (request is never global)
   ├─ 2. Notebook check           if notebook_id given:
   │                                it MUST belong to caller → else 404
   │
   ├─ 3. Query embedding          question → 768-dim vector (Gemini)
   │
   ├─ 4. Vector search (pgvector) — SQL-level filtering, never Python
   │     cosine_distance <=> query
   │     WHERE user_id      = caller's id        ← tenant isolation (mandatory)
   │       AND notebook_id  = notebook_id        ← ONLY when notebook_id was sent
   │     ORDER BY distance LIMIT top_k (default 5)
   │
   ├─ 5. Threshold filter         similarity = 1 − cosine_distance
   │                              keep chunks ≥ min_similarity (default 0.3)
   │
   ├─ 6. No chunks pass?          answer = "I could not find the answer…"
   │                              found_context: false — LLM never called
   │
   ├─ 7. Build context            labelled [Source 1..N] blocks
   │                              (capped at RAG_MAX_CONTEXT_CHARS = 12000)
   │
   └─ 8. Generate                 Gemini with "answer ONLY from context,
                                  cite [Source N]" instruction
             + retry with exponential backoff on transient errors
               (503 high demand, 429, 5xx...)
             + OpenRouter/Nemotron fallback if configured
             + think-blocks stripped, raw provider errors never shown

   Response: { answer, sources:[{snippet, document_id, ...}],
               found_context, model }
```

Notes:
- `user_id` filter is **always** applied in SQL — one user can never read
  another user's chunks, even by guessing notebook ids.
- The `notebook_id` filter is **only applied if the client sends one**.
  See section 3 — that "if" was historically the bug.
- WS streaming (`/api/chat/ws`) uses the exact same scoping, with the
  notebook id in the `start` frame and the same "must be owned" check.

---

## 3. Why answers used to come from ALL your files, not just the notebook

### What happened

You asked a question **inside Notebook A** and got answers grounded in
your Notebook B / C documents too.

### Root cause — the client never said which notebook it was in

The backend was already correct:

```python
# backend/app/services/retrieval.py  (_vector_search)
stmt = stmt.where(DocumentChunk.user_id == user_id)
if notebook_id is not None:                       # ← filter exists...
    stmt = stmt.where(DocumentChunk.notebook_id == notebook_id)
```

But the frontend page for notebooks called:

```ts
// frontend/src/pages/NotebookDetail.tsx  —  BEFORE
const res = await askQuestion(question)           // no notebook id sent!
```

`askQuestion()` in `frontend/src/services/api.ts` did not even have a
notebook parameter, so the request body had **no `notebook_id`**.

### The consequence chain

```
Frontend omits notebook_id
   → backend sets notebook_id = None
   → "if notebook_id is not None" filter is skipped
   → WHERE clause is only user_id = you
   → pgvector searches your ENTIRE library (every notebook)
   → top-5 most similar chunks overall win
   → sources list / [Source N] citations mix notebooks together
```

In other words, there was never a data leak — the `user_id` isolation held
at all times. The bug was purely **scope omission**: without an explicit
notebook filter, the explicit "my whole notebook library" search was the
default, and that is exactly right for the global Chat page but wrong when
you are standing inside a specific notebook.

### The fix (already applied)

```ts
// frontend/src/services/api.ts — askQuestion now accepts a notebook id
export async function askQuestion(
  question: string,
  topK?: number,
  minSimilarity?: number,
  notebookId?: number | null,   // ← NEW
): Promise<ChatResponse> {
  return request('/api/chat', {
    method: 'POST',
    body: { question, ..., ...(notebookId != null ? { notebook_id: notebookId } : {}) },
  })
}

// frontend/src/pages/NotebookDetail.tsx — AFTER
const res = await askQuestion(question, undefined, undefined, notebookId)
```

Now: inside a notebook → `notebook_id` flows all the way into the SQL
`WHERE notebook_id = ?`, and chunks from other notebooks can never enter
the context or the citations.

| Page                      | Scope after fix                                          |
|---------------------------|----------------------------------------------------------|
| Notebook detail page      | **Only that notebook's documents**                        |
| Global Chat / AI Insights | All of your notebooks (intended: "whole library" search)  |
| WS chat in Workspace      | All of your notebooks (`start` frame supports notebook_id if you scope it) |

---

## 4. Quick checklist if overlap ever reappears

1. Does the request body actually contain `notebook_id`?
   (browser devtools → Network → `/api/chat` payload)
2. Was the chunk uploaded *while* in that notebook?
   (`SELECT notebook_id, count(*) FROM document_chunks GROUP BY notebook_id;`)
3. Are you looking at the global Chat page (which intentionally searches all
   notebooks) or the notebook page?
4. If the answer says "could not find the answer...", the scoping worked but
   nothing in that notebook was similar enough (lower `min_similarity` or
   upload relevant docs — this message means *no cross-notebook fallback
   happens on purpose*).
