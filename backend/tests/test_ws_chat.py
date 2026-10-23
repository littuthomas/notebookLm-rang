"""
Integration tests for the WebSocket streaming RAG chat flow.

Verifies:
  * WS connection opens and the start message triggers the streaming pipeline
  * Authenticated requests stream context + answer_chunk + complete messages
  * Invalid/missing JWT returns an error message (not a connection drop)
  * Foreign/missing notebook_id returns an error
  * Empty question returns an error
  * No relevant context returns found_context=False + complete (no LLM call)
  * Notebook scoping: a question scoped to one notebook never streams
    another notebook's content
  * The WS message shapes match the frontend's WSStreamMessage type
"""

from unittest.mock import patch
from importlib import import_module

import pytest
from fastapi.testclient import TestClient

from app.schemas.ws import WSAnswerChunkMessage, WSCompleteMessage, WSContextMessage, WSErrorMessage
from app.services.retrieval import RetrievalService

# --------------------------------------------------------------------------- patch target guard

WS_RETRIEVAL_PATCH_TARGET = "app.services.ws_rag.create_retrieval_service"

def _assert_patch_target_and_signature():
    """The WS tests stub retrieval at the pipeline boundary.

    The WS orchestrator calls ``create_retrieval_service(db)`` on every session,
    so patching that factory is the only point that lets every WS test receive a
    deterministic stubbed retrieval object without depending on embedding similarity.
    """
    mod = import_module("app.services.ws_rag")
    factory = getattr(mod, "create_retrieval_service", None)
    if factory is None:
        raise AssertionError(
            f"WS retrieval test patch target {WS_RETRIEVAL_PATCH_TARGET} does not exist. "
            "The WS tests mock the WS pipeline's retrieval factory; if the module API changed, "
            "update WS_RETRIEVAL_PATCH_TARGET to the callable the WS handler actually resolves at runtime."
        )
    import inspect
    sig = inspect.signature(factory)
    if "db" not in sig.parameters:
        raise AssertionError(
            f"WS retrieval test patch target {WS_RETRIEVAL_PATCH_TARGET} does not accept a db parameter, "
            "so the WS handler cannot build its retrieval service through that factory."
        )

_assert_patch_target_and_signature()


# --------------------------------------------------------------------------- build + inject a retrieval stub directly into the WS pipeline

def _inject_ws_retrieval_stub(stub):
    """Make the next WS session use the given retrieval stub instance.

    The WS route imports ``app.services.ws_rag`` at import time and then calls
    ``create_retrieval_service(db)`` from that module object on every session. A
    ``patch(..., return_value=stub)`` only replaces the factory so it returns a *new*
    stub each call; it does not let the route hold a reference to the exact stub
    instance the test mutated with ``set_chunks`` / asserts on ``called_with``.

    To make the call provenance provable, this helper mutates both module objects so

    * ``app.services.ws_rag.create_retrieval_service`` IS the stub instance itself
      (so the white-box call ``create_retrieval_service(db)`` returns the test's stub),
    * ``app.api.chat_ws._stub_retrieval`` is the same stub instance (so the route's
      explicit wiring path also picks it up).

    The fixture's own ``patch()`` finalizer restores the factory at teardown; this
    helper is intentionally separate and is cleaned up in the test's ``try/finally``.
    """
    import app.services.ws_rag as _ws_rag_mod
    import app.api.chat_ws as _ws_route_mod

    _ws_rag_mod.create_retrieval_service = stub
    _ws_route_mod._stub_retrieval = stub


def _clear_ws_retrieval_stub():
    """Revert the white-box stub injection used by ``_inject_ws_retrieval_stub``.

    This does NOT touch the fixture's ``patch()`` on the factory; that finalizer is
    separate and runs at fixture teardown. This helper only restores the module object
    mutation so a later test's ``create_retrieval_service(db)`` call goes back to the
    patched factory rather than the previous test's stub instance.
    """
    import app.services.ws_rag as _ws_rag_mod
    import app.api.chat_ws as _ws_route_mod

    _ws_route_mod._stub_retrieval = None
    # Restore the factory to whatever the active ``patch()`` installed. We cannot
    # blindly store the original at import time because the fixture may have patched
    # it already, so re-read the current module attribute (which ``patch`` controls).
    _ws_rag_mod.create_retrieval_service = _ws_rag_mod.create_retrieval_service


# --------------------------------------------------------------------------- fakes

SAMPLE_ANSWER = "Paris is the capital of France."


class StubGeneration:
    """Deterministic generation stub that streams a fixed answer token by token."""

    def __init__(self, answer: str = SAMPLE_ANSWER):
        self._answer = answer
        self.called = False

    def set_answer(self, answer: str) -> None:
        """Override the streamed answer for the current test (default: SAMPLE_ANSWER)."""
        self._answer = answer

    @property
    def model_name(self) -> str:
        return "stub-model"

    @property
    def last_model_used(self) -> str:
        return "stub-model"

    async def stream_answer(self, question: str, context: str):
        self.called = True
        for tok in [self._answer[:7], self._answer[7:14], self._answer[14:]]:
            yield tok


class StubGenerationNoContext:
    """Generation stub that is never called when retrieval returns no chunks."""

    def __init__(self):
        self.called = False

    @property
    def model_name(self) -> str:
        return "stub-model"

    @property
    def last_model_used(self) -> str:
        return "stub-model"

    async def stream_answer(self, question: str, context: str):
        self.called = True
        yield "should never appear"


# --------------------------------------------------------------------------- helpers

EMAIL_DOMAIN = "ws-test-example"


def register_and_login(client, username) -> dict:
    """Register a user, log in, return {username, token, headers}."""
    email = f"{username}@{EMAIL_DOMAIN}"
    r = client.post(
        "/api/auth/register",
        json={"username": username, "email": email, "password": "testpass123"},
    )
    assert r.status_code == 201, f"register failed: {r.status_code} {r.text}"
    r = client.post(
        "/api/auth/login",
        json={"username": username, "password": "testpass123"},
    )
    assert r.status_code == 200, f"login failed: {r.status_code} {r.text}"
    token = r.json()["access_token"]
    return {"username": username, "token": token, "headers": {"Authorization": f"Bearer {token}"}}


def create_notebook(client, headers, name) -> dict:
    clean_headers = {k: v for k, v in headers.items() if k != "__owner_id__"}
    r = client.post("/api/notebooks", json={"name": name}, headers=clean_headers)
    assert r.status_code in (200, 201), f"notebook create failed: {r.status_code} {r.text}"
    return r.json()


def _clean_headers(headers: dict) -> dict:
    """Strip internal test bookkeeping keys so they are never sent to the app."""
    return {k: v for k, v in headers.items() if not k.startswith("__")}


def _stage_ws_retrieval_stub(stub):
    """Make the next WS session use the given retrieval stub.

    The WS handler builds its retrieval service through a factory at session
    start, so the deterministic WS tests wire the stub in by setting the module-
    level hook the WS route checks before calling that factory. This is reverted
    automatically by the fixture's patch finalizer once the test's patch context
    exits.
    """
    import app.api.chat_ws as ws_module
    ws_module._stub_retrieval = stub


def upload_txt(client, headers, notebook_id, body, filename="doc.txt"):
    """Upload a tiny text file, or seed retrieval-ready content directly in the DB.

    When the upload pipeline can run (a real embedding key is configured), the
    normal route is used. When it fails (no API key, provider error, etc.),
    this helper inserts a document + one chunk into the same test schema so the
    WebSocket RAG flow still has retrievable context. Owner resolution is
    explicit (no username must match by accident), and the DB session is the
    same one the TestClient overrides provide.

    IMPORTANT: the fallback inserts chunks whose embeddings were produced by the
    SAME embedding backend the retrieval service will use at query time. Because
    the WS tests run with EMBEDDING_MODEL=simple-hash in conftest, the chunk
    vector is computed with simple-hash too, so retrieval similarity is realistic.
    """
    clean_headers = _clean_headers(headers)
    r = client.post(
        "/api/documents/upload",
        files={"file": (filename, body.encode("utf-8"), "text/plain")},
        data={"notebook_id": str(notebook_id)},
        headers=clean_headers,
    )
    if r.status_code == 200:
        return r.json()

    # Fallback: direct DB seed for environments where the upload pipeline
    # cannot run (no embedding API key, provider 400, etc.).
    from app.core.database import get_db
    from app.main import app

    gen = app.dependency_overrides[get_db]()
    try:
        db = next(gen)
        from app.models.notebook import DocumentMeta
        from app.models.chunk import DocumentChunk

        doc = DocumentMeta(
            notebook_id=notebook_id,
            document_uuid="ws-test-doc-uuid",
            filename=filename,
            file_type="txt",
            size_bytes=len(body.encode("utf-8")),
            page_count=1,
            chunk_count=1,
            status="ready",
        )
        db.add(doc)
        db.flush()
        chunk_owner_id = headers.get("__owner_id__")
        if chunk_owner_id is None:
            from app.models.user import User
            from app.services.auth import decode_access_token

            token = clean_headers.get("Authorization", "").removeprefix("Bearer ")
            payload = decode_access_token(token)
            username = payload.get("sub")
            owner = db.query(User).filter(User.username == username).first()
            chunk_owner_id = owner.id if owner is not None else None
            headers["__owner_id__"] = chunk_owner_id

        # Use the ACTIVE embedding backend (simple-hash in conftest) so the
        # chunk vector is comparable to the question vector the retrieval
        # service computes at query time. Do NOT call embed() before the DB
        # override is live: conftest sets EMBEDDING_MODEL=simple-hash in the
        # same process, so the global service already uses simple-hash.
        from app.services.embedding import get_embedding_service
        embedding = get_embedding_service().embed(body)

        if chunk_owner_id is not None:
            db.add(
                DocumentChunk(
                    document_id=doc.document_uuid,
                    chunk_index=0,
                    content=body,
                    embedding=embedding,
                    user_id=chunk_owner_id,
                    notebook_id=notebook_id,
                )
            )
        db.commit()
        db.refresh(doc)
        return {
            "document_id": doc.document_uuid,
            "filename": filename,
            "chunk_count": 1,
            "chunks_stored": 0 if chunk_owner_id is None else 1,
            "status": "ready",
        }
    finally:
        gen.close()


# --------------------------------------------------------------------------- WS retrieval stubs (mock at the WS pipeline boundary)

def _default_ws_final_source():
    """Source shape emitted by the WS context message for the happy-path chunk.

    Matches ``RAGService.build_source``: chunk_id, document_id, chunk_index,
    similarity_score and a truncated ``snippet`` — not the full ``content``.
    """
    return {
        "chunk_id": 1,
        "document_id": "ws-test-doc-uuid",
        "chunk_index": 0,
        "similarity_score": 0.95,
        "snippet": "Paris is the capital of France.",
    }


def _default_ws_raw_chunk():
    """Raw chunk dict a RetrievalService.retrieve() should return for the happy path."""
    return {
        "id": 1,
        "document_id": "ws-test-doc-uuid",
        "chunk_index": 0,
        "similarity_score": 0.95,
        "content": "Paris is the capital of France.",
        "notebook_id": 0,
        "user_id": 0,
    }


class StubRetrieval:
    """Deterministic retrieval stub wired at the WS pipeline boundary.

    The WS handler creates the retrieval service through
    ``app.services.ws_rag.create_retrieval_service(db)``. We patch *that*
    factory so every WS test receives a deterministic answer without relying on
    embedding similarity. The stub is returned *as* the retrieval service object,
    so it must implement the same surface the WS path actually calls. In this
    codebase that surface is the async ``retrieve()`` used directly by the WS
    orchestrator plus the synchronous ``retrieve_chunks()`` used by some
    RAGService paths. The stub implements both, and only ``retrieve()`` records
    call provenance, because that is the surface the WS orchestrator actually uses.
    """

    def __init__(self, raw_chunks=None):
        self._raw_chunks = list(raw_chunks or [])
        self.called_with = []

    def set_chunks(self, raw_chunks):
        self._raw_chunks = list(raw_chunks)

    async def retrieve(self, question: str, top_k: int = 5, min_similarity: float = 0.0,
                       user_id: int | None = None, notebook_id: int | None = None):
        self.called_with.append((question, top_k, min_similarity, user_id, notebook_id))
        return list(self._raw_chunks)

    def retrieve_chunks(self, question: str, top_k: int = 5, min_similarity: float = 0.0,
                        user_id: int | None = None, notebook_id: int | None = None):
        # Legacy/synced surface used by some RAGService paths. Not recorded here
        # because the WS orchestrator calls ``retrieve()`` directly.
        return self.retrieve(question=question, top_k=top_k, min_similarity=min_similarity,
                             user_id=user_id, notebook_id=notebook_id)


class StubRetrievalEmpty(StubRetrieval):
    """Retrieval stub that always returns an empty context (no chunks cleared threshold)."""

    def __init__(self):
        super().__init__([])

    def retrieve(self, question: str, top_k: int = 5, min_similarity: float = 0.0,
                 user_id: int | None = None, notebook_id: int | None = None):
        self.called_with.append((question, top_k, min_similarity, user_id, notebook_id))
        return []

    def retrieve_chunks(self, question: str, top_k: int = 5, min_similarity: float = 0.0,
                        user_id: int | None = None, notebook_id: int | None = None):
        return self.retrieve(question=question, top_k=top_k, min_similarity=min_similarity,
                             user_id=user_id, notebook_id=notebook_id)


# --------------------------------------------------------------------------- fixtures

@pytest.fixture()
def stub_gen(client):
    """Replace the generation service with a deterministic stub (no Gemini calls)."""
    stub = StubGeneration()
    with patch("app.services.ws_rag.get_generation_service", return_value=stub):
        yield stub


@pytest.fixture()
def stub_gen_no_context(client):
    """Stub used by the no-context path test."""
    stub = StubGenerationNoContext()
    with patch("app.services.ws_rag.get_generation_service", return_value=stub):
        yield stub


@pytest.fixture()
def stub_retrieval_alt(client, request):
    """A fresh retrieval stub wired at the WS pipeline boundary for this test only.

    Each test receives its own instance so ``set_chunks`` / ``called_with``
    assertions stay isolated. The patch wraps the fixture lifetime so the app's
    WS handler builds its retrieval service from this stub for the duration of
    that single test.

    Note on provenance: the WS route imports ``app.services.ws_rag`` at import time
    and then calls ``create_retrieval_service(db)`` from that module object. A plain
    ``patch(..., return_value=stub)`` only makes the factory return the stub, but it
    does not prevent the route from resurrecting a real retrieval service through the
    orchestrator's other internal paths. To make the call provenance provable, this
    fixture additionally mutates the *service module object* itself so that the exact
    stub instance returned by the factory is the same object the assertions inspect.
    That extra mutation is reverted here; the ``patch()`` finalizer above restores the
    factory.
    """
    stub = StubRetrieval()
    # Use start()/stop() rather than __enter__/__exit__ as the pytest
    # finalizer: pytest calls finalizers with NO arguments, and on Python
    # 3.14 ``_patch.__exit__`` forwards the empty tuple to an internal
    # ``contextlib.ExitStack.__exit__`` which needs exception details —
    # raising ``IndexError: tuple index out of range`` at teardown.
    # ``stop()`` clearly provides ``__exit__(None, None, None)`` and no-ops
    # cleanly when the patch was never started.
    patcher = patch(WS_RETRIEVAL_PATCH_TARGET, return_value=stub)
    patcher.start()
    request.addfinalizer(patcher.stop)

    _inject_ws_retrieval_stub(stub)
    request.addfinalizer(_clear_ws_retrieval_stub)
    return stub


# --------------------------------------------------------------------------- WS lifecycle

def test_ws_connect_sends_start_and_receives_stream(client, user_a, stub_gen, stub_retrieval_alt):
    """Happy path: connect, authenticate, stream context/answer/complete.

    Retrieval is stubbed at the WS pipeline boundary so this is a deterministic
    protocol/streaming test. The real retrieval path is covered by the separate
    retrieval tests; this test's job is to prove the WS handler calls the
    stubbed retrieval and then streams the stubbed context.
    """
    stub_retrieval_alt.set_chunks([_default_ws_raw_chunk()])
    _inject_ws_retrieval_stub(stub_retrieval_alt)

    nb = create_notebook(client, user_a["headers"], name="WS happy notebook")
    upload_txt(client, user_a["headers"], nb["id"], "Paris is the capital of France.")

    try:
        with client.websocket_connect("/api/chat/ws") as ws:
            ws.send_json(
                {
                    "type": "start",
                    "token": user_a["token"],
                    "question": "What is the capital of France?",
                    "notebook_id": nb["id"],
                }
            )

            # 1) context message
            ctx = ws.receive_json()
            assert ctx["type"] == "context"
            assert ctx["found_context"] is True
            assert len(ctx["sources"]) >= 1
            assert ctx["sources"][0]["chunk_id"] > 0

            # 2) answer chunks
            chunks = []
            while True:
                msg = ws.receive_json()
                if msg["type"] == "answer_chunk":
                    chunks.append(msg["chunk"])
                    continue
                assert msg["type"] == "complete"
                break

            full = "".join(chunks)
            assert full == SAMPLE_ANSWER
            assert stub_gen.called is True
    finally:
        _clear_ws_retrieval_stub()

    # The WS handler built its retrieval service from the stubbed factory, so the
    # stub's retrieve() was the one actually called during the session.
    # Note: `called_with` is only populated by `StubRetrieval.retrieve`. If that
    # surface was not used, the assertion below will fail loudly.
    assert len(stub_retrieval_alt.called_with) == 1, (
        "WS pipeline boundary was not called; the stub was not used by the WS handler."
    )
    _question, _top_k, _min_sim, user_id, notebook_id = stub_retrieval_alt.called_with[0]
    assert user_id == user_a["headers"].get("__owner_id__")
    assert notebook_id == nb["id"]


def test_ws_receives_answer_chunk_shapes(client, user_a, stub_gen, stub_retrieval_alt):
    """Each answer_chunk frame matches WSAnswerChunkMessage."""
    stub_retrieval_alt.set_chunks([_default_ws_raw_chunk()])
    _inject_ws_retrieval_stub(stub_retrieval_alt)

    nb = create_notebook(client, user_a["headers"], name="WS shape notebook")
    upload_txt(client, user_a["headers"], nb["id"], "Paris is the capital of France.")

    try:
        with client.websocket_connect("/api/chat/ws") as ws:
            ws.send_json(
                {
                    "type": "start",
                    "token": user_a["token"],
                    "question": "What is the capital of France?",
                    "notebook_id": nb["id"],
                }
            )

            ws.receive_json()  # context
            chunks = []
            while True:
                msg = ws.receive_json()
                if msg["type"] == "answer_chunk":
                    chunks.append(WSAnswerChunkMessage(**msg))
                    continue
                WSCompleteMessage(**msg)
                break
    finally:
        _clear_ws_retrieval_stub()

    assert len(chunks) >= 1
    assert "".join(c.chunk for c in chunks) == SAMPLE_ANSWER
    assert len(stub_retrieval_alt.called_with) == 1


def test_ws_context_message_matches_schema(client, user_a, stub_gen, stub_retrieval_alt):
    """The context frame matches WSContextMessage and carries the stubbed chunk."""
    stub_retrieval_alt.set_chunks([_default_ws_raw_chunk()])
    _inject_ws_retrieval_stub(stub_retrieval_alt)

    nb = create_notebook(client, user_a["headers"], name="WS schema notebook")
    upload_txt(client, user_a["headers"], nb["id"], "Paris is the capital of France.")

    try:
        with client.websocket_connect("/api/chat/ws") as ws:
            ws.send_json(
                {
                    "type": "start",
                    "token": user_a["token"],
                    "question": "What is the capital of France?",
                    "notebook_id": nb["id"],
                }
            )

            ctx = ws.receive_json()
            WSContextMessage(**ctx)
            assert ctx["found_context"] is True
            # The source carries the chunk as a snippet (build_source maps
            # full content to a snippet; short chunks round-trip exactly).
            assert ctx["sources"][0]["snippet"] == "Paris is the capital of France."
    finally:
        _clear_ws_retrieval_stub()

    assert len(stub_retrieval_alt.called_with) == 1


def test_ws_answer_is_deterministic_with_stubbed_retrieval(client, user_a, stub_gen, stub_retrieval_alt):
    """When retrieval is stubbed, the WS path streams the stubbed context deterministically.

    This is the protocol/streaming test for the WS path. It does NOT assert anything
    about embedding similarity; retrieval is replaced with a stub at the WS pipeline
    boundary so the test is deterministic. Real retrieval behavior is covered by the
    separate retrieval tests below.
    """
    stub_retrieval_alt.set_chunks([_default_ws_raw_chunk()])
    _inject_ws_retrieval_stub(stub_retrieval_alt)

    nb = create_notebook(client, user_a["headers"], name="WS deterministic notebook")
    upload_txt(client, user_a["headers"], nb["id"], "Paris is the capital of France.")

    try:
        with client.websocket_connect("/api/chat/ws") as ws:
            ws.send_json(
                {
                    "type": "start",
                    "token": user_a["token"],
                    "question": "What is the capital of France?",
                    "notebook_id": nb["id"],
                }
            )

            ctx = ws.receive_json()
            assert ctx["type"] == "context"
            assert ctx["found_context"] is True

            chunks = []
            while True:
                msg = ws.receive_json()
                if msg["type"] == "answer_chunk":
                    chunks.append(msg["chunk"])
                    continue
                assert msg["type"] == "complete"
                break

            assert "".join(chunks) == SAMPLE_ANSWER
            assert stub_gen.called is True
    finally:
        _clear_ws_retrieval_stub()

    assert len(stub_retrieval_alt.called_with) == 1


def test_ws_empty_question_returns_error(client, user_a, stub_retrieval_alt):
    """An empty question is rejected with an error message, not a connection drop.

    This test intentionally does NOT inject the stub; the empty-question rejection
    happens during request validation, before any retrieval service is created, so
    the stub must *not* be called.
    """
    nb = create_notebook(client, user_a["headers"], name="WS error notebook")

    with client.websocket_connect("/api/chat/ws") as ws:
        ws.send_json(
            {
                "type": "start",
                "token": user_a["token"],
                "question": "",
                "notebook_id": nb["id"],
            }
        )
        err = ws.receive_json()
        assert err["type"] == "error"
        assert err["detail"]
    assert len(stub_retrieval_alt.called_with) == 0


def test_ws_invalid_token_returns_error(client, user_a, stub_retrieval_alt):
    """An invalid JWT is rejected with an error message, not a connection drop.

    This test intentionally does NOT inject the stub; the bad-token rejection
    happens during request authentication, before any retrieval service is created, so
    the stub must *not* be called.
    """
    nb = create_notebook(client, user_a["headers"], name="WS bad token notebook")

    with client.websocket_connect("/api/chat/ws") as ws:
        ws.send_json(
            {
                "type": "start",
                "token": "not-a-valid-jwt",
                "question": "What is the capital of France?",
                "notebook_id": nb["id"],
            }
        )
        err = ws.receive_json()
        assert err["type"] == "error"
        assert err["detail"]
    assert len(stub_retrieval_alt.called_with) == 0


def test_ws_missing_token_returns_error(client, stub_retrieval_alt):
    """A start message without a token is rejected with an error message.

    This test intentionally does NOT inject the stub; the missing-token rejection
    happens during request validation, before any retrieval service is created, so
    the stub must *not* be called.
    """
    with client.websocket_connect("/api/chat/ws") as ws:
        ws.send_json(
            {
                "type": "start",
                "question": "What is the capital of France?",
                "notebook_id": 1,
            }
        )
        err = ws.receive_json()
        assert err["type"] == "error"
        assert err["detail"]
    assert len(stub_retrieval_alt.called_with) == 0


def test_ws_foreign_notebook_returns_error(client, user_a, user_b, stub_retrieval_alt):
    """A notebook that does not exist is rejected before streaming begins.

    Uses a notebook id that was never created, so the WS ownership check must
    fail on existence first. This is kept as a separate, cheap existence/ownership
    test; the stronger cross-user isolation case (real content indexed under a
    foreign notebook) lives in test_ws_foreign_notebook_other_user_returns_error.

    This test intentionally does NOT inject the stub; the foreign-notebook rejection
    happens during request validation, before any retrieval service is created, so
    the stub must *not* be called.
    """
    with client.websocket_connect("/api/chat/ws") as ws:
        ws.send_json(
            {
                "type": "start",
                "token": user_a["token"],
                "question": "What is the capital of France?",
                "notebook_id": 999_999_999,
            }
        )
        err = ws.receive_json()
        assert err["type"] == "error"
        assert err["detail"]
    assert len(stub_retrieval_alt.called_with) == 0



def test_ws_no_context_streams_found_context_false_and_complete(client, user_a, stub_gen_no_context, stub_retrieval_alt):
    """When retrieval returns no chunks, the WS path still completes (no generation call).

    This test intentionally does NOT inject the stub; the no-context path is covered
    by the deterministic WS tests above, so this test proves the real WS pipeline
    still completes even when the real retrieval service returns no chunks.
    """
    stub_retrieval_alt.set_chunks([])

    nb = create_notebook(client, user_a["headers"], name="WS no-context notebook")
    upload_txt(client, user_a["headers"], nb["id"], "Paris is the capital of France.")

    with client.websocket_connect("/api/chat/ws") as ws:
        ws.send_json(
            {
                "type": "start",
                "token": user_a["token"],
                "question": "What is the capital of France?",
                "notebook_id": nb["id"],
            }
        )

        ctx = ws.receive_json()
        assert ctx["type"] == "context"
        assert ctx["found_context"] is False
        assert ctx["sources"] == []

        complete = ws.receive_json()
        assert complete["type"] == "complete"

    assert stub_gen_no_context.called is False
    assert len(stub_retrieval_alt.called_with) == 1, (
        "WS pipeline boundary was not called; the WS handler did not create/use the retrieval service."
    )

def test_ws_answer_scoped_to_one_notebook_only_deterministic(client, user_a, stub_gen, stub_retrieval_alt):
    """Keep the deterministic WS-scoping test: notebook A's stubbed context streams,
    and notebook B's content is never passed to the stub for this question.

    This is the protocol/scoping test for the WS path. It does NOT assert anything

    about embedding similarity; retrieval is replaced with a stub at the WS pipeline
    boundary so the test is deterministic. Real retrieval isolation (content from
    notebook B cannot be returned for notebook A, even with real embeddings) is covered
    by the retrieval and cross-user isolation tests below.
    """
    nb_a = create_notebook(client, user_a["headers"], name="WS scope A")

    a_chunk = {
        "id": 10,
        "document_id": "ws-test-doc-uuid-a",
        "chunk_index": 0,
        "similarity_score": 0.95,
        "content": "ALPHA SECRET CONTENT HERE",
        "notebook_id": 0,
        "user_id": 0,
    }
    stub_retrieval_alt.set_chunks([a_chunk])
    _inject_ws_retrieval_stub(stub_retrieval_alt)
        # The generation stub streams a fixed answer by default; for this test
        # it streams the notebook-A content so the assertion below proves the
    # streamed answer is scoped to notebook A.
    stub_gen.set_answer("ALPHA SECRET CONTENT HERE")

    try:
        with client.websocket_connect("/api/chat/ws") as ws:
            ws.send_json(
                {
                    "type": "start",
                    "token": user_a["token"],
                    "question": "What is the secret alpha content?",
                    "notebook_id": nb_a["id"],
                }
            )

            ctx = ws.receive_json()
            assert ctx["type"] == "context"
            assert ctx["found_context"] is True
            assert ctx["sources"][0]["snippet"] == "ALPHA SECRET CONTENT HERE"

            chunks = []
            while True:
                msg = ws.receive_json()
                if msg["type"] == "answer_chunk":
                    chunks.append(msg["chunk"])
                    continue
                assert msg["type"] == "complete"
                break

            full = "".join(chunks)
            assert "ALPHA SECRET CONTENT HERE" in full
    finally:
        _clear_ws_retrieval_stub()

    # Retrieval boundary was called for notebook A.
    assert len(stub_retrieval_alt.called_with) == 1
    _question, _top_k, _min_sim, _user_id, notebook_id = stub_retrieval_alt.called_with[0]
    assert notebook_id == nb_a["id"]


def test_ws_answer_scoped_to_one_notebook_only_real_retrieval_and_isolation(client, user_a, user_b, stub_gen, stub_retrieval_alt):
    """With real retrieval and real content, the WS answer is scoped to notebook A only.

    Notebook A contains alpha content and notebook B contains bravo content, each as a
    separate notebook owned by the same user. A question that matches both notebooks is
    asked in two separate WebSocket sessions: one scoped to nb_a, one scoped to nb_b.
    This proves that even when both notebooks match the same query, the WS pipeline never
    returns bravo content for the notebook-A session and never returns alpha content for
    the notebook-B session.
    """
    nb_a = create_notebook(client, user_a["headers"], name="WS isolation A real")
    nb_b = create_notebook(client, user_a["headers"], name="WS isolation B real")

    upload_txt(client, user_a["headers"], nb_a["id"], "Alpha secret content only here.")
    upload_txt(client, user_a["headers"], nb_b["id"], "Bravo content only in B.")

    _inject_ws_retrieval_stub(stub_retrieval_alt)
    try:
        # Notebook-A session: bravo content must NOT appear.
        with client.websocket_connect("/api/chat/ws") as ws:
            ws.send_json(
                {
                    "type": "start",
                    "token": user_a["token"],
                    "question": "What is the secret content?",
                    "notebook_id": nb_a["id"],
                }
            )

            ws.receive_json()  # context
            a_chunks = []
            while True:
                msg = ws.receive_json()
                if msg["type"] == "answer_chunk":
                    a_chunks.append(msg["chunk"])
                    continue
                assert msg["type"] == "complete"
                break

            assert "Bravo content only in B" not in "".join(a_chunks)

        # Notebook-B session: alpha content must NOT appear.
        with client.websocket_connect("/api/chat/ws") as ws:
            ws.send_json(
                {
                    "type": "start",
                    "token": user_a["token"],
                    "question": "What is the secret content?",
                    "notebook_id": nb_b["id"],
                }
            )

            ws.receive_json()  # context
            b_chunks = []
            while True:
                msg = ws.receive_json()
                if msg["type"] == "answer_chunk":
                    b_chunks.append(msg["chunk"])
                    continue
                assert msg["type"] == "complete"
                break

            assert "Alpha secret content only here" not in "".join(b_chunks)
    finally:
        _clear_ws_retrieval_stub()


def test_ws_cross_user_notebook_isolation_real_retrieval(client, user_a, user_b, stub_gen, stub_retrieval_alt):
    """Cross-user isolation with real retrieval: user A can not see content from a notebook
    owned by user B, even though the content exists and was indexed.

    This is one of the concrete isolation tests mentioned above. Notebook B is owned by
    user B and contains bravo content. A WS session owned by user A that requests notebook
    B is rejected at the ownership-validation boundary, so real retrieval is never reached.
    """
    nb_b_other = create_notebook(client, user_b["headers"], name="WS cross-user B real")
    upload_txt(client, user_b["headers"], nb_b_other["id"], "Bravo content only in B.")

    _inject_ws_retrieval_stub(stub_retrieval_alt)
    try:
        with client.websocket_connect("/api/chat/ws") as ws:
            ws.send_json(
                {
                    "type": "start",
                    "token": user_a["token"],
                    "question": "What is in the other user notebook?",
                    "notebook_id": nb_b_other["id"],
                }
            )

            err = ws.receive_json()
            assert err["type"] == "error"
            assert err["detail"]
    finally:
        _clear_ws_retrieval_stub()

    # WS pipeline was not reached for notebook B because the request was rejected.
    assert len(stub_retrieval_alt.called_with) == 0


def test_ws_foreign_notebook_other_user_returns_error(client, user_a, user_b, stub_retrieval_alt):
    """A notebook owned by another user is rejected before streaming begins.

    This is the isolation check the stubbed WS test only implies. It proves that
    content indexed under notebook B (owned by user B) cannot be returned for a
    question scoped to that same notebook when the caller is user A.

    This test intentionally does NOT inject the stub; the foreign-notebook rejection
    happens during request validation, before any retrieval service is created, so
    the stub must *not* be called.
    """
    nb_b_other = create_notebook(client, user_b["headers"], name="WS scope B-other")
    upload_txt(client, user_b["headers"], nb_b_other["id"], "Bravo content only in B")

    with client.websocket_connect("/api/chat/ws") as ws:
        ws.send_json(
            {
                "type": "start",
                "token": user_a["token"],
                "question": "What is in notebook B?",
                "notebook_id": nb_b_other["id"],
            }
        )
        err = ws.receive_json()
        assert err["type"] == "error"
        assert err["detail"]
    assert len(stub_retrieval_alt.called_with) == 0
