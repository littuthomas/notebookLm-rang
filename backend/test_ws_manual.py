"""Manual WebSocket test. Run from backend/ directory.

Usage:
    python test_ws_manual.py "What is the capital of France?"
    python test_ws_manual.py "Who was the first person to walk on Mars?" --min-similarity 0.7
"""
import asyncio
import json
import sys
import httpx
import websockets


async def run(question: str, min_similarity: float | None = None):
    base = "http://127.0.0.1:8002"

    # --- Step 1: Get a JWT token ---
    print("1. Logging in...")
    async with httpx.AsyncClient() as http:
        r = await http.post(f"{base}/api/auth/register", json={
            "username": "manual_test", "email": "manual@test.com", "password": "test1234"
        })
        if r.status_code not in (201, 409):
            print(f"   Register failed: {r.status_code} {r.text}")
            return
        r = await http.post(f"{base}/api/auth/login", json={
            "username": "manual_test", "password": "test1234"
        })
        token = r.json()["access_token"]

        # Chat requires an owned notebook scope now: create/reuse one.
        r = await http.post(
            f"{base}/api/notebooks",
            json={"name": "Manual WS test notebook"},
            headers={"Authorization": f"Bearer {token}"},
        )
        if r.status_code not in (200, 201):
            print(f"   Notebook creation failed: {r.status_code} {r.text}")
            return
        notebook_id = r.json()["id"]
        print(f"   Notebook ready (id={notebook_id})")
    print(f"   Token obtained ({len(token)} chars)")

    # --- Step 2: Connect and send question ---
    msg = {"type": "start", "token": token, "question": question, "notebook_id": notebook_id}
    if min_similarity is not None:
        msg["min_similarity"] = min_similarity

    print(f"\n2. Connecting to ws://127.0.0.1:8002/api/chat/ws ...")
    print(f"   Sending: {json.dumps(msg, indent=2)}\n")

    async with websockets.connect("ws://127.0.0.1:8002/api/chat/ws") as ws:
        await ws.send(json.dumps(msg))

        # --- Step 3: Receive messages ---
        print("3. Server messages:\n")
        full_answer = ""
        while True:
            raw = await ws.recv()
            data = json.loads(raw)
            t = data.get("type")

            if t == "context":
                found = data.get("found_context")
                sources = data.get("sources", [])
                print(f"   [context]  found_context={found}  sources={len(sources)}")
                for s in sources:
                    print(f"             - {s.get('document_id')}: sim={s.get('similarity_score')} "
                          f"\"{s.get('snippet', '')[:60]}...\"")
            elif t == "answer_chunk":
                chunk = data.get("chunk", "")
                full_answer += chunk
                print(f"   [chunk]    \"{chunk}\"")
            elif t == "complete":
                print(f"   [complete]")
                break
            elif t == "error":
                print(f"   [error]    {data.get('detail')}")
                break
            else:
                print(f"   [???]      {data}")

    print(f"\n4. Final answer: \"{full_answer}\"")
    print("   Done.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    q = sys.argv[1]
    ms = None
    if "--min-similarity" in sys.argv:
        idx = sys.argv.index("--min-similarity")
        ms = float(sys.argv[idx + 1])
    asyncio.run(run(q, ms))
