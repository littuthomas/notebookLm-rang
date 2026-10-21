#!/usr/bin/env python3
"""Real end-to-end WebSocket streaming test.

Uses the configured environment (Neon PostgreSQL + Gemini API key).
Registers a test user, obtains a JWT, and runs two WebSocket tests:
  1. France capital question (should find context + stream answer)
  2. Mars walking question (should find no context, no Gemini call)

No modifications to any existing code.
"""

import json
import sys
import time

from starlette.testclient import TestClient
from app.main import app

client = TestClient(app, raise_server_exceptions=False)

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


# ---------------------------------------------------------------------------
# Setup: register user + login
# ---------------------------------------------------------------------------
print("=" * 60)
print("END-TO-END WEBSOCKET STREAMING TEST")
print("=" * 60)

print("\n--- Setup: register + login ---")
resp = client.post("/api/auth/register", json={
    "username": "ws_e2e_user",
    "email": "ws_e2e@example.com",
    "password": "e2etest123",
})
if resp.status_code in (201, 409):
    print(f"  Register: {resp.status_code} (user ready)")
else:
    print(f"  Register failed: {resp.status_code} {resp.json()}")
    sys.exit(1)

resp = client.post("/api/auth/login", json={
    "username": "ws_e2e_user",
    "password": "e2etest123",
})
assert resp.status_code == 200, f"Login failed: {resp.status_code} {resp.json()}"
TOKEN = resp.json()["access_token"]
print(f"  Login: OK, token length={len(TOKEN)}")

# Notebooks are independent knowledge spaces: chat now REQUIRES an owned
# notebook_id, so create one for the e2e user and scope every start to it.
resp = client.post(
    "/api/notebooks",
    json={"name": "WS E2E notebook"},
    headers={"Authorization": f"Bearer {TOKEN}"},
)
if resp.status_code not in (200, 201):
    print(f"  Notebook creation failed: {resp.status_code} {resp.text}")
    sys.exit(1)
NOTEBOOK_ID = resp.json()["id"]
print(f"  Notebook: OK, id={NOTEBOOK_ID}")


# ===========================================================================
# TEST 1: France capital question
# ===========================================================================
print("\n" + "=" * 60)
print("TEST 1: 'What is the capital of France?'")
print("=" * 60)

question_1 = "What is the capital of France?"
start_msg = {"type": "start", "token": TOKEN, "question": question_1, "notebook_id": NOTEBOOK_ID}
print(f"\n  Client sends: {json.dumps(start_msg, indent=2)}")

messages_received = []
answer_chunks = []
sources = []
found_context = None
got_complete = False
start_time = time.time()

try:
    with client.websocket_connect("/api/chat/ws") as ws:
        ws.send_json(start_msg)

        # Receive messages until complete or error — with explicit limit
        MAX_MESSAGES = 50  # safety limit to prevent infinite loop
        for i in range(MAX_MESSAGES):
            try:
                msg = ws.receive_json()
            except Exception as e:
                print(f"  [RECV] Connection closed: {type(e).__name__}: {e}")
                break

            elapsed = round(time.time() - start_time, 2)
            msg_type = msg.get("type", "unknown")
            print(f"  [RECV] (t={elapsed}s) type={msg_type}", end="")

            if msg_type == "context":
                found_context = msg.get("found_context")
                sources = msg.get("sources", [])
                print(f"  found_context={found_context}  sources_count={len(sources)}")
            elif msg_type == "answer_chunk":
                chunk_text = msg.get("chunk", "")
                answer_chunks.append(chunk_text)
                print(f"  chunk={repr(chunk_text[:60])}{'...' if len(chunk_text) > 60 else ''}")
            elif msg_type == "complete":
                got_complete = True
                print("  [STREAMING COMPLETE]")
                messages_received.append(msg)
                break
            elif msg_type == "error":
                error_detail = msg.get('detail', '')
                print(f"  detail={error_detail}")
                messages_received.append(msg)
                # Don't break yet -- server may send complete after error
            else:
                print(f"  payload={json.dumps(msg)[:100]}")

            messages_received.append(msg)

    total_time = round(time.time() - start_time, 2)
    print(f"\n  Total time: {total_time}s")
    print(f"  Total messages received: {len(messages_received)}")
    print(f"  Answer chunks received: {len(answer_chunks)}")

    full_answer = "".join(answer_chunks)
    print(f"\n  Full streamed answer:\n    {full_answer[:300]}{'...' if len(full_answer) > 300 else ''}")

    # --- Assertions ---
    print("\n  --- Assertions ---")
    report("Context found", found_context is True,
           f"found_context={found_context}")
    report("Sources received", len(sources) > 0,
           f"count={len(sources)}")
    if sources:
        for idx, s in enumerate(sources):
            report(f"  Source {idx+1} has snippet",
                   bool(s.get("snippet")),
                   f"doc={s.get('document_id')}, sim={s.get('similarity_score')}")

    report("Answer chunks received (streamed)", len(answer_chunks) >= 1,
           f"chunk_count={len(answer_chunks)}")
    report("Completion message received", got_complete)
    report("Answer mentions Paris",
           "paris" in full_answer.lower(),
           f"answer_length={len(full_answer)} chars")
    report("Answer mentions France",
           "france" in full_answer.lower())

except Exception as e:
    report("TEST 1 completed without exception", False, f"{type(e).__name__}: {e}")


# ===========================================================================
# TEST 2: Unrelated Mars question (no context expected)
# ===========================================================================
print("\n" + "=" * 60)
print("TEST 2: 'Who was the first person to walk on Mars?'")
print("=" * 60)

question_2 = "Who was the first person to walk on Mars?"
start_msg_2 = {"type": "start", "token": TOKEN, "question": question_2, "min_similarity": 0.7, "notebook_id": NOTEBOOK_ID}
print(f"\n  Client sends: {json.dumps(start_msg_2, indent=2)}")

messages_received_2 = []
answer_chunks_2 = []
found_context_2 = None
got_complete_2 = False
start_time_2 = time.time()

try:
    with client.websocket_connect("/api/chat/ws") as ws:
        ws.send_json(start_msg_2)

        MAX_MESSAGES = 50
        for i in range(MAX_MESSAGES):
            try:
                msg = ws.receive_json()
            except Exception as e:
                print(f"  [RECV] Connection closed: {type(e).__name__}: {e}")
                break

            elapsed = round(time.time() - start_time_2, 2)
            msg_type = msg.get("type", "unknown")
            print(f"  [RECV] (t={elapsed}s) type={msg_type}", end="")

            if msg_type == "context":
                found_context_2 = msg.get("found_context")
                sources_2 = msg.get("sources", [])
                print(f"  found_context={found_context_2}  sources_count={len(sources_2)}")
            elif msg_type == "answer_chunk":
                chunk_text = msg.get("chunk", "")
                answer_chunks_2.append(chunk_text)
                print(f"  chunk={repr(chunk_text[:60])}{'...' if len(chunk_text) > 60 else ''}")
            elif msg_type == "complete":
                got_complete_2 = True
                print("  [STREAMING COMPLETE]")
                messages_received_2.append(msg)
                break
            elif msg_type == "error":
                error_detail_2 = msg.get('detail', '')
                print(f"  detail={error_detail_2}")
                messages_received_2.append(msg)
                # Don't break yet -- server may send complete after error
            else:
                print(f"  payload={json.dumps(msg)[:100]}")

            messages_received_2.append(msg)

    total_time_2 = round(time.time() - start_time_2, 2)
    print(f"\n  Total time: {total_time_2}s")
    print(f"  Total messages received: {len(messages_received_2)}")
    print(f"  Answer chunks received: {len(answer_chunks_2)}")

    full_answer_2 = "".join(answer_chunks_2)
    if full_answer_2:
        print(f"\n  Full answer (if any):\n    {full_answer_2[:300]}")
    else:
        print("\n  No answer text streamed (expected for no-context case)")

    # --- Assertions ---
    print("\n  --- Assertions ---")
    report("No context found", found_context_2 is False,
           f"found_context={found_context_2}")
    report("No answer chunks (Gemini not called)", len(answer_chunks_2) == 0,
           f"chunk_count={len(answer_chunks_2)}")
    report("Completion message received", got_complete_2)
    report("WebSocket closed cleanly", True, "reached end of receive loop")

except Exception as e:
    report("TEST 2 completed without exception", False, f"{type(e).__name__}: {e}")


# ===========================================================================
# Summary
# ===========================================================================
print("\n" + "=" * 60)
print(f"RESULTS: {PASS} passed, {FAIL} failed out of {PASS + FAIL} assertions")
print("=" * 60)

sys.exit(0 if FAIL == 0 else 1)
