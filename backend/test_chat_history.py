#!/usr/bin/env python3
"""End-to-end tests for the chat-history endpoints (Stage 2).

Uses the configured environment (Neon PostgreSQL) like test_ws_e2e.py.
Registers two throwaway users, then verifies:
  1. Session creation (POST /api/chat/sessions)
  2. Listing (GET /api/chat/sessions) - only the caller's sessions
  3. Direct message recording + title derivation (service-level)
  4. Detail view (GET /api/chat/sessions/{id}) with ordered messages
  5. Ownership: user B gets 404 for user A's session (get + delete)
  6. Unauthenticated access gets 401
  7. Optional session_id persistence via POST /api/chat
  8. DELETE removes the session and cascades messages

Cleans up its test sessions afterwards (messages cascade automatically).
Requires the backend's .env and a reachable database.
No modifications to any existing code.
"""

import sys
import uuid

from fastapi import FastAPI
from starlette.testclient import TestClient

TAG = "chat_hist_e2e"

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
    suffix = f" -- {detail}" if detail else ""
    print(f"  [{tag}] {name}{suffix}")


def expect(cond: bool, name: str, detail: str = "") -> bool:
    report(name, cond, detail)
    return cond


# ---------------------------------------------------------------------------
# Setup: two isolated users
# ---------------------------------------------------------------------------
print("=" * 60)
print("CHAT HISTORY END-TO-END TESTS (Stage 2)")
print("=" * 60)

# Build a minimal ASGI app with ONLY the routes we need, so the test never
# touches Gemini/embeddings/RAG. Session persistence via POST /api/chat is
# covered separately below with a stubbed RAG service.
from app.api.auth import router as auth_router  # noqa: E402
from app.api.chat_history import router as history_router  # noqa: E402

test_app = FastAPI()
test_app.include_router(auth_router)
test_app.include_router(history_router)
client = TestClient(test_app, raise_server_exceptions=False)

suffix = uuid.uuid4().hex[:8]
USER_A = f"{TAG}_a_{suffix}"
USER_B = f"{TAG}_b_{suffix}"
EMAIL_A = f"{USER_A}@example.com"
EMAIL_B = f"{USER_B}@example.com"
PASSWORD = "e2etest123"

created_session_ids: list[int] = []


def register_and_login(username: str, email: str) -> dict:
    resp = client.post(
        "/api/auth/register",
        json={"username": username, "email": email, "password": PASSWORD},
    )
    if resp.status_code not in (201, 409):
        print(f"Setup register failed: {resp.status_code} {resp.text}")
        sys.exit(1)

    resp = client.post(
        "/api/auth/login",
        json={"username": username, "password": PASSWORD},
    )
    if resp.status_code != 200:
        print(f"Setup login failed: {resp.status_code} {resp.text}")
        sys.exit(1)
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


print(f"\n--- Setup: users {USER_A} / {USER_B} ---")
headers_a = register_and_login(USER_A, EMAIL_A)
headers_b = register_and_login(USER_B, EMAIL_B)
print("  [OK] both users ready")

# ---------------------------------------------------------------------------
# 1. Create session
# ---------------------------------------------------------------------------
print("\n--- 1. Create session ---")
resp = client.post("/api/chat/sessions", headers=headers_a)
if not expect(resp.status_code == 201, "POST /api/chat/sessions -> 201", str(resp.status_code)):
    sys.exit(1)
body = resp.json()
session_a = body["id"]
created_session_ids.append(session_a)
expect(body["title"] == "New chat", "new session titled 'New chat'", body["title"])
expect(body["message_count"] == 0, "new session has 0 messages")

# ---------------------------------------------------------------------------
# 2. Unauthenticated access is rejected
# ---------------------------------------------------------------------------
print("\n--- 2. Unauthenticated access ---")
expect(client.post("/api/chat/sessions").status_code == 401, "create without token -> 401")
expect(client.get("/api/chat/sessions").status_code == 401, "list without token -> 401")
expect(client.get(f"/api/chat/sessions/{session_a}").status_code == 401, "get without token -> 401")
expect(client.delete(f"/api/chat/sessions/{session_a}").status_code == 401, "delete without token -> 401")

# ---------------------------------------------------------------------------
# 3. Ownership: user B cannot see A's session
# ---------------------------------------------------------------------------
print("\n--- 3. Ownership isolation ---")
resp = client.get(f"/api/chat/sessions/{session_a}", headers=headers_b)
expect(resp.status_code == 404, "B GET A's session -> 404", str(resp.status_code))
resp = client.delete(f"/api/chat/sessions/{session_a}", headers=headers_b)
expect(resp.status_code == 404, "B DELETE A's session -> 404", str(resp.status_code))
resp = client.get("/api/chat/sessions", headers=headers_b)
expect(resp.json()["total"] == 0, "B's list does not contain A's session")

# ---------------------------------------------------------------------------
# 4. Service-level: record exchanges + title derivation (user A)
# ---------------------------------------------------------------------------
print("\n--- 4. Record exchanges + title derivation ---")
from app.core.database import get_db  # noqa: E402
from app.services.chat_history import create_chat_history_service, derive_title  # noqa: E402

db = next(get_db())
try:
    from app.models.user import User  # noqa: E402

    user_a = db.query(User).filter(User.username == USER_A).first()

    # Title derivation unit checks (pure function, no DB writes)
    expect(derive_title("What is RAG?") == "What is RAG?", "short question -> exact title")
    long_q = (
        "Explain the retrieval augmented generation pipeline step by step "
        "including embedding, vector search and answer generation please"
    )
    t = derive_title(long_q)
    expect(len(t) <= 61, "long question title <= ~60 chars", f"len={len(t)}")
    expect(t.endswith("…"), "truncated title ends with ellipsis")

    svc = create_chat_history_service(db)
    saved = svc.record_exchange(
        session_id=session_a,
        user_id=user_a.id,
        question=long_q,
        answer="RAG retrieves relevant chunks and grounds the answer in them.",
    )
    if expect(saved is not None, "record_exchange saved both messages"):
        expect(saved[0].role == "user" and saved[1].role == "assistant", "roles stored as user/assistant")

    svc.record_exchange(
        session_id=session_a,
        user_id=user_a.id,
        question="Second question: how does pgvector rank results?",
        answer="By cosine similarity between the question and chunk embeddings.",
    )
    db.expire_all()

    # Foreign/missing session is rejected at the service boundary too
    user_b = db.query(User).filter(User.username == USER_B).first()
    foreign_user_id = user_b.id if user_b else -1
    expect(
        svc.record_exchange(session_id=session_a, user_id=foreign_user_id, question="x", answer="y") is None,
        "record_exchange rejects another user's session",
    )
    expect(
        svc.record_exchange(session_id=999999, user_id=user_a.id, question="x", answer="y") is None,
        "record_exchange rejects a missing session",
    )
finally:
    db.close()

# ---------------------------------------------------------------------------
# 5. Detail view: messages ordered, title derived from first question
# ---------------------------------------------------------------------------
print("\n--- 5. Session detail ---")
resp = client.get(f"/api/chat/sessions/{session_a}", headers=headers_a)
expect(resp.status_code == 200, "GET session detail -> 200", str(resp.status_code))
detail = resp.json()
msgs = detail["messages"]
expect(detail["title"].startswith("Explain the retrieval"), "title derived from first question", detail["title"])
expect(len(msgs) == 4, "4 messages stored (2 exchanges)", str(len(msgs)))
if len(msgs) == 4:
    expect(
        [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"],
        "messages ordered user/assistant/user/assistant",
        str([m["role"] for m in msgs]),
    )
    expect(msgs[0]["content"].startswith("Explain the retrieval"), "first message is the first question")

# ---------------------------------------------------------------------------
# 6. List view: newest activity first, message_count correct
# ---------------------------------------------------------------------------
print("\n--- 6. Session list ---")
resp = client.post("/api/chat/sessions", headers=headers_a)
expect(resp.status_code == 201, "A created a second session")
session_a2 = resp.json()["id"]
created_session_ids.append(session_a2)

resp = client.get("/api/chat/sessions", headers=headers_a)
lst = resp.json()
expect(lst["total"] == 2, "A sees 2 sessions", str(lst["total"]))
by_id = {s["id"]: s for s in lst["sessions"]}
expect(by_id[session_a]["message_count"] == 4, "message_count == 4 for session A", str(by_id[session_a]["message_count"]))
expect(by_id[session_a2]["message_count"] == 0, "empty second session has count 0")
# Both sessions exist; the just-created session_a2 has the newest updated_at,
# so "newest activity first" legitimately puts it at the top.
expect(lst["sessions"][0]["id"] == session_a2, "newest activity listed first", str(lst["sessions"][0]["id"]))

# ---------------------------------------------------------------------------
# 7. POST /api/chat persistence (additive session_id) - stubbed RAG
# ---------------------------------------------------------------------------
print("\n--- 7. POST /api/chat with session_id (stubbed RAG) ---")
from unittest.mock import patch  # noqa: E402

from app.api import chat as chat_api  # noqa: E402
from app.schemas.chat import ChatRequest  # noqa: E402

# ChatRequest accepts optional session_id
req = ChatRequest(question="Does optional session_id parse?", session_id=session_a2)
expect(req.session_id == session_a2, "ChatRequest.session_id accepted")
expect(ChatRequest(question="no session").session_id is None, "ChatRequest.session_id defaults to None")

full_app = FastAPI()
full_app.include_router(auth_router)
full_app.include_router(history_router)
from app.api.chat import router as chat_router  # noqa: E402

full_app.include_router(chat_router)
full_client = TestClient(full_app, raise_server_exceptions=False)

fake_result = {
    "answer": "Stubbed answer for history persistence.",
    "sources": [],
    "found_context": True,
    "model": "stub-model",
}
with patch.object(chat_api, "create_rag_service") as factory:
    factory.return_value.answer_question.return_value = fake_result
    resp = full_client.post(
        "/api/chat",
        json={"question": "What is stored in history?", "session_id": session_a2},
        headers=headers_a,
    )
expect(resp.status_code == 200, "POST /api/chat -> 200 with stubbed RAG", f"{resp.status_code} {resp.text[:200]}")
expect(resp.json()["answer"] == fake_result["answer"], "answer came from the RAG service")

resp = full_client.get(f"/api/chat/sessions/{session_a2}", headers=headers_a)
detail2 = resp.json()
expect(detail2["title"] == "What is stored in history?", "title set from first REST question", detail2["title"])
expect(len(detail2["messages"]) == 2, "Q&A persisted via REST", str(len(detail2["messages"])))
if len(detail2["messages"]) == 2:
    expect(detail2["messages"][0]["content"] == "What is stored in history?", "user question stored verbatim")
    expect(detail2["messages"][1]["content"] == fake_result["answer"], "assistant answer stored verbatim")

# Wrong owner on session_id: answer still succeeds, nothing is stored
with patch.object(chat_api, "create_rag_service") as factory:
    factory.return_value.answer_question.return_value = fake_result
    resp = full_client.post(
        "/api/chat",
        json={"question": "B tries to write into A's session", "session_id": session_a},
        headers=headers_b,
    )
expect(resp.status_code == 200, "POST /api/chat still succeeds with foreign session_id")
resp = full_client.get(f"/api/chat/sessions/{session_a}", headers=headers_a)
expect(len(resp.json()["messages"]) == 4, "foreign write attempt stored nothing (still 4 messages)")
expect(resp.json()["title"].startswith("Explain the retrieval"), "A's title unchanged")

# Omitted session_id behaves exactly as before: answer only, nothing stored
with patch.object(chat_api, "create_rag_service") as factory:
    factory.return_value.answer_question.return_value = fake_result
    resp = full_client.post("/api/chat", json={"question": "No history please"}, headers=headers_a)
expect(resp.status_code == 200, "POST /api/chat without session_id -> 200 (backward compatible)")
resp = full_client.get(f"/api/chat/sessions/{session_a2}", headers=headers_a)
expect(len(resp.json()["messages"]) == 2, "no extra messages stored when session_id omitted")

# ---------------------------------------------------------------------------
# 8. DELETE with cascade
# ---------------------------------------------------------------------------
print("\n--- 8. Delete session (cascade) ---")
resp = client.delete(f"/api/chat/sessions/{session_a2}", headers=headers_a)
expect(resp.status_code == 200, "DELETE own session -> 200", str(resp.status_code))
if resp.status_code == 200:
    expect(resp.json()["messages_deleted"] == 2, "delete reports 2 messages removed", resp.text[:120])
expect(client.get(f"/api/chat/sessions/{session_a2}", headers=headers_a).status_code == 404, "deleted session -> 404")
created_session_ids.remove(session_a2)

resp = client.get("/api/chat/sessions", headers=headers_a)
expect(resp.json()["total"] == 1, "A has 1 session left")

# ---------------------------------------------------------------------------
# Cleanup: remove test sessions and users (messages cascade with sessions)
# ---------------------------------------------------------------------------
print("\n--- Cleanup ---")
from app.models.chat import ChatMessage, ChatSession  # noqa: E402
from app.models.user import User as UserModel  # noqa: E402

db2 = next(get_db())
try:
    removed_sessions = (
        db2.query(ChatSession).filter(ChatSession.id.in_(created_session_ids)).delete(synchronize_session=False)
    )
    for username in (USER_A, USER_B):
        u = db2.query(UserModel).filter(UserModel.username == username).first()
        if u:
            db2.delete(u)  # chat_sessions cascade via ORM relationship
    db2.commit()
    leftovers = (
        db2.query(ChatMessage)
        .filter(ChatMessage.session_id.in_(created_session_ids or [-1]))
        .count()
    )
    expect(leftovers == 0, "no orphan chat_messages after cleanup")
    print(f"  Removed {removed_sessions} test session(s) and both test users")
except Exception as e:
    db2.rollback()
    print(f"  [WARN] cleanup issue: {e}")
finally:
    db2.close()

# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print(f"RESULTS: {PASS} passed, {FAIL} failed")
print("=" * 60)
sys.exit(1 if FAIL else 0)
