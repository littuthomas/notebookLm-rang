#!/usr/bin/env python3
"""Stage 3 tests: WebSocket chat-history persistence + session ownership.

Uses the full app (real Neon DB) with a stubbed WSRAGOrchestrator so no
Gemini/retrieval calls are made. Verifies:
  1. Invalid JWT -> auth error (protocol preserved)
  2. Happy path: session_id + full stream -> exactly one user message and
     one complete assistant message stored AFTER complete
  3. Partial stream (orchestrator fails mid-stream) -> nothing stored
  4. No-context stream (context + complete, no chunks) -> nothing stored
  5. Omitted session_id -> nothing stored (protocol unchanged)
  6. Foreign session_id -> nothing stored, streaming unaffected
  7. Persistence failure -> stream unaffected, error only logged
  8. REST chat-history endpoints still behave (regression spot-check)

Cleans up its test users/sessions afterwards (messages cascade).
Requires the backend's .env and a reachable database.
"""

import sys
import uuid
from unittest.mock import patch

from fastapi import FastAPI
from starlette.testclient import TestClient

from app.main import app

PASS = 0
FAIL = 0


def report(name: str, ok: bool, detail: str = ""):
    global PASS, FAIL
    if ok:
        PASS += 1
        tag = "OK"
    else:
        FAIL += 1
        tag = "FAIL"
    print(f"  [{tag}] {name}{f' -- {detail}' if detail else ''}")


def expect(cond: bool, name: str, detail: str = "") -> bool:
    report(name, cond, detail)
    return cond


print("=" * 60)
print("WEBSOCKET CHAT HISTORY TESTS (Stage 3)")
print("=" * 60)

TAG = "ws_hist_e2e"
suffix = uuid.uuid4().hex[:8]
USER_A = f"{TAG}_a_{suffix}"
USER_B = f"{TAG}_b_{suffix}"
PASSWORD = "e2etest123"

client = TestClient(app, raise_server_exceptions=False)


def register_and_login(username: str) -> dict:
    resp = client.post(
        "/api/auth/register",
        json={"username": username, "email": f"{username}@example.com", "password": PASSWORD},
    )
    if resp.status_code not in (201, 409):
        print(f"Setup register failed: {resp.status_code} {resp.text}")
        sys.exit(1)
    resp = client.post("/api/auth/login", json={"username": username, "password": PASSWORD})
    if resp.status_code != 200:
        print(f"Setup login failed: {resp.status_code} {resp.text}")
        sys.exit(1)
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


print(f"\n--- Setup: users {USER_A} / {USER_B} ---")
headers_a = register_and_login(USER_A)
headers_b = register_and_login(USER_B)
token_a = headers_a["Authorization"].split(" ", 1)[1]
token_b = headers_b["Authorization"].split(" ", 1)[1]
print("  [OK] both users ready")

# Sessions for A and B
session_a = client.post("/api/chat/sessions", headers=headers_a).json()["id"]
session_b = client.post("/api/chat/sessions", headers=headers_b).json()["id"]
print(f"  [OK] sessions: A={session_a} B={session_b}")


def ws_start(token: str, question: str, session_id: int | None = None) -> dict:
    msg = {"type": "start", "token": token, "question": question}
    if session_id is not None:
        msg["session_id"] = session_id
    return msg


def run_ws(start_msg: dict, stream: list[dict] | Exception):
    """Open a WS, send start, and drive a stubbed orchestrator stream."""
    outer = {}

    def fake_orchestrator(db, generation_service=None):
        outer["db"] = db

        class FakeOrch:
            async def handle_question(self, **kwargs):
                if isinstance(stream, Exception):
                    # Raise BEFORE yielding anything, like a real orchestrator
                    # failing at retrieval/generation start.
                    raise stream
                for item in stream:
                    if isinstance(item, Exception):
                        # Raise mid-stream, like a real generator failure.
                        raise item
                    yield item

        return FakeOrch()

    with patch("app.api.chat_ws.WSRAGOrchestrator", fake_orchestrator):
        with client.websocket_connect("/api/chat/ws") as ws:
            ws.send_json(start_msg)
            received = []
            while True:
                frame = ws.receive_json()
                received.append(frame)
                if frame.get("type") in ("complete", "error"):
                    break
    # The orchestrator (and its db handle) only exists when the flow got past
    # authentication; auth-failure runs legitimately have no db.
    return received, outer.get("db")


HAPPY_STREAM = [
    {"type": "context", "found_context": True, "sources": [{"id": 1, "document_id": "d", "chunk_index": 0, "content": "c", "similarity_score": 0.9}]},
    {"type": "answer_chunk", "chunk": "RAG retrieves "},
    {"type": "answer_chunk", "chunk": "relevant chunks."},
    {"type": "complete"},
]


def messages_of(db, session_id: int):
    from app.models.chat import ChatMessage

    return (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at, ChatMessage.id)
        .all()
    )


# ---------------------------------------------------------------------------
print("\n--- 1. Invalid JWT still errors before anything else ---")
received, _ = run_ws(ws_start("not-a-jwt", "q"), HAPPY_STREAM)
expect(received[-1]["type"] == "error" and "Authentication failed" in received[-1]["detail"], "invalid JWT -> error frame")

# ---------------------------------------------------------------------------
print("\n--- 2. Happy path: full stream + owned session -> 2 messages stored ---")
received, db = run_ws(ws_start(token_a, "What is RAG?", session_a), HAPPY_STREAM)
expect([f["type"] for f in received] == ["context", "answer_chunk", "answer_chunk", "complete"], "protocol frames unchanged", str([f["type"] for f in received]))
db.expire_all()
msgs = messages_of(db, session_a)
expect(len(msgs) == 2, "exactly 2 messages stored", str(len(msgs)))
if len(msgs) == 2:
    expect(msgs[0].role == "user" and msgs[0].content == "What is RAG?", "user message stored verbatim")
    expect(msgs[1].role == "assistant" and msgs[1].content == "RAG retrieves relevant chunks.", "assistant message is the joined full answer", msgs[1].content)

# ---------------------------------------------------------------------------
print("\n--- 3. Mid-stream failure -> nothing stored ---")
mid_fail = RuntimeError("stream broke")
received, db = run_ws(ws_start(token_a, "Failing question", session_a), [HAPPY_STREAM[0], {"type": "answer_chunk", "chunk": "partial"}, mid_fail])
expect(received[-1]["type"] == "error", "mid-stream failure -> error frame")
db.expire_all()
expect(len(messages_of(db, session_a)) == 2, "no partial data stored (still 2 messages)")

# ---------------------------------------------------------------------------
print("\n--- 4. No-context flow (context + complete, no chunks) -> nothing stored ---")
no_ctx = [{"type": "context", "found_context": False, "sources": []}, {"type": "complete"}]
received, db = run_ws(ws_start(token_a, "No context question", session_a), no_ctx)
expect([f["type"] for f in received] == ["context", "complete"], "no-context protocol preserved")
db.expire_all()
expect(len(messages_of(db, session_a)) == 2, "nothing stored without an answer (still 2 messages)")

# ---------------------------------------------------------------------------
print("\n--- 5. Omitted session_id -> nothing stored ---")
received, db = run_ws(ws_start(token_a, "No history question"), HAPPY_STREAM)
expect(received[-1]["type"] == "complete", "stream completes normally without session_id")
db.expire_all()
expect(len(messages_of(db, session_a)) == 2, "no messages stored when session_id omitted")

# ---------------------------------------------------------------------------
print("\n--- 6. Foreign session_id -> nothing stored, stream unaffected ---")
received, db = run_ws(ws_start(token_b, "B writes into A?", session_a), HAPPY_STREAM)
expect(received[-1]["type"] == "complete", "B's stream still completes", str(received[-1]))
db.expire_all()
expect(len(messages_of(db, session_a)) == 2, "A's session untouched by B (still 2 messages)")
expect(len(messages_of(db, session_b)) == 0, "B's own session also untouched")

# ---------------------------------------------------------------------------
print("\n--- 7. Persistence failure -> stream unaffected ---")
def failing_record(self, **kwargs):
    raise RuntimeError("db exploded")

with patch("app.services.chat_history.ChatHistoryService.record_exchange", failing_record):
    received, _ = run_ws(ws_start(token_a, "Persist fail question", session_a), HAPPY_STREAM)
expect(received[-1]["type"] == "complete", "answer still completes when history save fails", str(received[-1]))
expect(any(f["type"] == "answer_chunk" for f in received), "all answer chunks still delivered")

# ---------------------------------------------------------------------------
print("\n--- 8. REST chat-history regression spot-check ---")
resp = client.get(f"/api/chat/sessions/{session_a}", headers=headers_a)
expect(resp.status_code == 200, "GET session detail -> 200")
expect(resp.json()["title"] == "What is RAG?", "title derived from first WS question", resp.json()["title"])
expect(len(resp.json()["messages"]) == 2, "detail view shows the 2 WS-stored messages")
resp = client.get("/api/chat/sessions", headers=headers_a)
expect(resp.status_code == 200 and resp.json()["total"] == 1, "list shows A's single session")
resp = client.delete(f"/api/chat/sessions/{session_a}", headers=headers_a)
expect(resp.status_code == 200 and resp.json()["messages_deleted"] == 2, "delete cascades the WS messages")
client.delete(f"/api/chat/sessions/{session_b}", headers=headers_b)

# ---------------------------------------------------------------------------
print("\n--- Cleanup ---")
from app.core.database import get_db  # noqa: E402
from app.models.user import User as UserModel  # noqa: E402

db2 = next(get_db())
try:
    removed = 0
    for username in (USER_A, USER_B):
        u = db2.query(UserModel).filter(UserModel.username == username).first()
        if u:
            db2.delete(u)  # sessions/messages cascade via ORM relationship
            removed += 1
    db2.commit()
    print(f"  Removed {removed} test user(s) with their sessions/messages")
except Exception as e:
    db2.rollback()
    print(f"  [WARN] cleanup issue: {e}")
finally:
    db2.close()

print("\n" + "=" * 60)
print(f"RESULTS: {PASS} passed, {FAIL} failed")
print("=" * 60)
sys.exit(1 if FAIL else 0)
