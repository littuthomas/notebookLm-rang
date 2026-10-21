#!/usr/bin/env python3
"""Deterministic WebSocket tests for the streaming RAG endpoint.

These tests verify:
  1. Empty-question handling (Pydantic validation error → error + close)
  2. Invalid-JWT handling (auth failure → error + close)
  3. POST /api/chat endpoint is still registered and reachable
  4. The application starts and imports successfully

No real Gemini calls are made; no database is required for the WS error-path tests.
"""

import json
import sys
import threading
import time
import urllib.request
import urllib.error

from starlette.testclient import TestClient

# ---------------------------------------------------------------------------
# Import the app (this also verifies all imports succeed)
# ---------------------------------------------------------------------------
from app.main import app

client = TestClient(app, raise_server_exceptions=False)

PASS = 0
FAIL = 0


def report(name: str, ok: bool, detail: str = ""):
    global PASS, FAIL
    if ok:
        PASS += 1
        status = "OK"
    else:
        FAIL += 1
        status = "FAIL"
    suffix = f" — {detail}" if detail else ""
    print(f"  {status}: {name}{suffix}")


# ===== Test 1: Empty-question handling =====================================

print("\n--- Test 1: Empty-question handling ---")
try:
    with client.websocket_connect("/api/chat/ws") as ws:
        ws.send_json({"type": "start", "token": "dummy-token", "question": ""})

        # The server should respond with an error (Pydantic min_length=1 fails)
        msg1 = ws.receive_json()
        report("First message is error", msg1.get("type") == "error",
               f"got type={msg1.get('type')}")
        report("Error mentions validation/empty",
               "Invalid start message" in msg1.get("detail", "") or
               "Question cannot be empty" in msg1.get("detail", ""),
               f"detail={msg1.get('detail', '')[:80]}")

        # Server should close the connection after the error (we used `break`)
        # The next receive should raise (WebSocketDisconnect or similar)
        try:
            msg2 = ws.receive_json()
            # If we get here, check it's a close frame or error
            report("Connection closed after error",
                   msg2.get("type") == "error" or msg2 is None,
                   f"unexpected message: {msg2}")
        except Exception:
            # WebSocketDisconnect or similar — this is expected
            report("Connection closed after error", True, "WebSocketDisconnect raised as expected")

except Exception as e:
    report("Empty-question test completed", False, f"Exception: {e}")


# ===== Test 2: Invalid-JWT handling ========================================

print("\n--- Test 2: Invalid-JWT handling ---")
try:
    with client.websocket_connect("/api/chat/ws") as ws:
        ws.send_json({
            "type": "start",
            "token": "definitely-not-a-valid-jwt",
            "question": "What is the capital of France?",
            # notebook_id is now mandatory; a bogus JWT fails before the
            # notebook ownership check runs, so any id keeps exercising the
            # intended auth-failure path.
            "notebook_id": 1,
        })

        # Server should respond with an auth error
        msg1 = ws.receive_json()
        report("First message is error", msg1.get("type") == "error",
               f"got type={msg1.get('type')}")
        report("Error mentions authentication",
               "Authentication failed" in msg1.get("detail", "") or
               "Invalid" in msg1.get("detail", ""),
               f"detail={msg1.get('detail', '')[:80]}")

        # Server should close the connection
        try:
            msg2 = ws.receive_json()
            report("Connection closed after auth error",
                   msg2.get("type") in ("error", "complete") or msg2 is None,
                   f"unexpected message: {msg2}")
        except Exception:
            report("Connection closed after auth error", True, "WebSocketDisconnect raised as expected")

except Exception as e:
    report("Invalid-JWT test completed", False, f"Exception: {e}")


# ===== Test 3: POST /api/chat endpoint still registered =====================

print("\n--- Test 3: POST /api/chat endpoint registration ---")
try:
    # Check OpenAPI schema for the /api/chat path
    resp = client.get("/openapi.json")
    schema = resp.json()
    paths = list(schema.get("paths", {}).keys())
    report("OpenAPI schema returns paths", len(paths) > 0, f"paths={paths}")
    report("/api/chat POST registered",
           "/api/chat" in paths and "post" in schema["paths"].get("/api/chat", {}),
           f"methods in /api/chat: {list(schema['paths'].get('/api/chat', {}).keys())}")
except Exception as e:
    report("POST /api/chat check", False, f"Exception: {e}")


# ===== Test 4: App starts and health check works ===========================

print("\n--- Test 4: App starts and health check ---")
try:
    resp = client.get("/")
    report("Root endpoint returns 200", resp.status_code == 200,
           f"status={resp.status_code}")
    body = resp.json()
    report("Root endpoint has message", "message" in body,
           f"keys={list(body.keys())}")

    resp = client.get("/health")
    report("Health endpoint returns 200", resp.status_code == 200,
           f"status={resp.status_code}")
except Exception as e:
    report("App startup check", False, f"Exception: {e}")


# ===== Test 5: WebSocket /api/chat/ws endpoint is in OpenAPI (optional) =====

print("\n--- Test 5: WebSocket endpoint presence ---")
try:
    resp = client.get("/openapi.json")
    schema = resp.json()
    # WebSocket endpoints may not appear in OpenAPI, but we can verify the route exists
    # by checking that the app has the websocket route
    routes = [r.path for r in app.routes if hasattr(r, 'path')]
    report("WebSocket route /api/chat/ws registered",
           "/api/chat/ws" in routes,
           f"ws routes: {[r for r in routes if 'ws' in r]}")
except Exception as e:
    report("WebSocket endpoint check", False, f"Exception: {e}")


# ===== Summary ==============================================================

print("\n" + "=" * 60)
print(f"RESULTS: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
print("=" * 60)

sys.exit(0 if FAIL == 0 else 1)
