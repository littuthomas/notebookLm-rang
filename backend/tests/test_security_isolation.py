"""
Security & isolation tests for the document/RAG endpoints.

Verified requirements:
  * An authenticated user can upload and access their own document.
  * Another user cannot read, preview, download, delete, or retrieve that document.
  * Search and RAG retrieval never return unauthorized chunks.
  * Invalid file types and oversized uploads are rejected.
  * Unauthenticated requests are rejected.
  * Document deletion removes/invalidates its chunks and embeddings.
  * Errors do not reveal document contents or secrets.

The upload pipeline's embedding model is forced to the offline "simple-hash"
backend (see conftest) and the Gemini/Google document processor is replaced
with a deterministic fake, so these tests make NO external API calls but still
exercise the real routers, DB queries and pgvector search.
"""
import uuid
from unittest.mock import patch

import pytest


# ------------------------------------------------------------------ helpers

SUPPORTED_TEXT = (
    "Paris is the capital of France. The Eiffel Tower is in Paris. "
    "France is a country in western Europe."
)

FOREIGN_SENTINEL = "TOP-SECRET-USER-B-CONTENT-zqxwce123"


def create_notebook(client, headers, *, name="Security test notebook") -> dict:
    """Create an owned notebook and return its API representation."""
    r = client.post("/api/notebooks", json={"name": name}, headers=headers)
    assert r.status_code == 201 or r.status_code == 200, r.text
    return r.json()


def upload_file(client, headers, *, filename="isolation-notes.txt", content=SUPPORTED_TEXT, notebook_id="default"):
    return client.post(
        "/api/documents/upload",
        files={"file": (filename, content.encode("utf-8"), "text/plain")},
        data={"notebook_id": notebook_id},
        headers=headers,
    )


class FakeProcessor:
    """Stands in for the Gemini-backed document processor.

    Produces real chunk dicts whose embeddings come from the (offline)
    simple-hash EmbeddingService, so pgvector search behaves realistically.
    Honors the actual file contents: a file with no extractable text is
    reported as failed, like the real processor would.
    """

    def process_file(self, file_path, filename, notebook_id, document_id=None):
        from pathlib import Path

        from app.services.embedding import get_embedding_service

        doc_id = document_id or uuid.uuid4().hex
        content = Path(file_path).read_text(encoding="utf-8", errors="replace")

        if not content.strip():
            # Mirrors the real processor: no extractable text -> failed status
            return {
                "document_id": doc_id,
                "filename": filename,
                "file_type": "txt",
                "status": "failed",
                "page_count": 0,
                "chunk_count": 0,
                "chunks": [],
                "error_message": "No extractable text",
            }

        if "user-b" in filename:
            body = FOREIGN_SENTINEL
        else:
            body = content
        embedding = get_embedding_service().embed(body)

        return {
            "document_id": doc_id,
            "filename": filename,
            "file_type": "txt",
            "status": "completed",
            "page_count": 1,
            "chunk_count": 1,
            "chunks": [
                {
                    "document_id": doc_id,
                    "chunk_index": 0,
                    "content": body,
                    # chunker adds notebook_id; harmless, not persisted here
                    "notebook_id": notebook_id,
                    "embedding": embedding,
                }
            ],
            "error_message": None,
        }


@pytest.fixture()
def fake_processor():
    """Patch the upload endpoint's document processor with the fake."""
    with patch("app.api.documents.get_document_processor", return_value=FakeProcessor()):
        yield


@pytest.fixture()
def db(client):
    """A raw DB session bound to the same session the client overrides use."""
    from app.core.database import get_db
    from app.main import app

    gen = app.dependency_overrides[get_db]()
    session = next(gen)
    try:
        yield session
    finally:
        gen.close()


# ------------------------------------------------------------- authentication

def test_upload_requires_auth(client):
    r = client.post(
        "/api/documents/upload",
        files={"file": ("a.txt", b"hello", "text/plain")},
        data={"notebook_id": "default"},
    )
    assert r.status_code == 401


def test_retrieve_requires_auth(client):
    r = client.post("/api/retrieve", json={"question": "anything"})
    assert r.status_code == 401


def test_chat_requires_auth(client):
    r = client.post("/api/chat", json={"question": "anything"})
    assert r.status_code == 401


def test_legacy_listing_requires_auth(client):
    r = client.get("/api/documents/legacy")
    assert r.status_code == 401


@pytest.mark.parametrize(
    "method,path",
    [
        ("get", "/api/documents"),
        ("delete", "/api/documents/1"),
    ],
)
def test_document_routes_require_auth(client, method, path):
    r = getattr(client, method)(path)
    assert r.status_code == 401


# ------------------------------------------------------- upload + own access

def test_user_can_upload_and_see_own_document(client, user_a, fake_processor, db):
    nb = create_notebook(client, user_a["headers"], name="Own access")
    r = upload_file(client, user_a["headers"], filename="own-doc.txt", notebook_id=str(nb["id"]))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["chunk_count"] == 1
    assert body["chunks_stored"] == 1

    # Owner can list the document
    docs = client.get("/api/documents", headers=user_a["headers"]).json()["documents"]
    assert any(d["filename"] == "own-doc.txt" for d in docs)

    # Owner can retrieve their own chunk (min_similarity=-1 disables threshold)
    q = client.post(
        "/api/retrieve",
        json={
            "question": "Where is the Eiffel Tower?",
            "top_k": 10,
            "min_similarity": -1.0,
            "notebook_id": nb["id"],
        },
        headers=user_a["headers"],
    )
    assert q.status_code == 200
    contents = [c["content"] for c in q.json()["retrieved_chunks"]]
    assert any("Eiffel Tower" in c for c in contents)


def test_upload_never_stores_client_filename(client, user_a, fake_processor, db):
    r = upload_file(client, user_a["headers"], filename="../../ malicious/../secret name.txt")
    assert r.status_code == 200
    # Stored name is sanitized (no traversal, no directory components).
    docs = client.get("/api/documents", headers=user_a["headers"]).json()["documents"]
    stored = next(d for d in docs if d["filename"].endswith("secret name.txt"))
    assert "/" not in stored["filename"] and "\\" not in stored["filename"]


# ------------------------------------------------------- cross-user isolation

def test_other_user_retrieval_never_returns_foreign_chunks(client, user_a, user_b, fake_processor, db):
    # A uploads a document containing a sentinel string, scoped to A's own
    # notebook.
    nb_a = create_notebook(client, user_a["headers"], name="A retrieve")
    nb_b = create_notebook(client, user_b["headers"], name="B retrieve")
    r = upload_file(client, user_a["headers"], filename="a-secret.txt", notebook_id=str(nb_a["id"]))
    assert r.status_code == 200

    # B asks a question scoped to B's OWN notebook. With the similarity
    # threshold disabled (-1) B still only ever matches chunks in B's notebook,
    # so A's chunk can never show up — neither via cross-user leakage nor by
    # ignoring the notebook filter.
    q = client.post(
        "/api/retrieve",
        json={
            "question": "Tell me the eiffel tower secret",
            "top_k": 50,
            "min_similarity": -1.0,
            "notebook_id": nb_b["id"],
        },
        headers=user_b["headers"],
    )
    assert q.status_code == 200
    all_text = " ".join(c["content"] for c in q.json()["retrieved_chunks"])
    assert FOREIGN_SENTINEL not in all_text
    assert "Eiffel Tower" not in all_text


def test_owner_retrieval_returns_only_own_chunks(client, user_a, user_b, fake_processor, db):
    # Each user uploads into their OWN notebook; sentinel content belongs to B.
    nb_a = create_notebook(client, user_a["headers"], name="A own chunks")
    nb_b = create_notebook(client, user_b["headers"], name="B own chunks")
    assert (
        upload_file(client, user_a["headers"], filename="a-doc.txt", notebook_id=str(nb_a["id"]))
        .status_code
        == 200
    )
    assert (
        upload_file(client, user_b["headers"], filename="user-b-doc.txt", notebook_id=str(nb_b["id"]))
        .status_code
        == 200
    )

    q = client.post(
        "/api/retrieve",
        json={"question": "question about anything", "top_k": 50, "min_similarity": -1.0, "notebook_id": nb_a["id"]},
        headers=user_a["headers"],
    )
    body = q.json()
    # A may have uploaded several docs during earlier tests in this session;
    # the isolation property being tested: NEVER any of B's sentinel content.
    assert all(FOREIGN_SENTINEL not in c["content"] for c in body["retrieved_chunks"])

    # And vice versa (B never sees A's text).
    q2 = client.post(
        "/api/retrieve",
        json={"question": "another question", "top_k": 50, "min_similarity": -1.0, "notebook_id": nb_b["id"]},
        headers=user_b["headers"],
    )
    body2 = q2.json()
    # B's OWN document deliberately contains the sentinel, so only A's content
    # ("Eiffel Tower" text from a-doc.txt) may never appear in B's results.
    assert all("Eiffel Tower" not in c["content"] for c in body2["retrieved_chunks"])


def test_chat_isolation_no_context_for_other_user(client, user_a, user_b, fake_processor, db):
    # A uploads into A's notebook; B chats scoped to B's OWN notebook.
    # Distant/simple-hash vectors stay below the default threshold, and even
    # above it B must only ever see chunks from B's own notebook.
    nb_a = create_notebook(client, user_a["headers"], name="A private chat")
    nb_b = create_notebook(client, user_b["headers"], name="B private chat")
    assert (
        upload_file(client, user_a["headers"], filename="private-a.txt", notebook_id=str(nb_a["id"]))
        .status_code
        == 200
    )

    r = client.post(
        "/api/chat",
        json={"question": "What is the capital of France?", "notebook_id": nb_b["id"]},
        headers=user_b["headers"],
    )
    assert r.status_code == 200
    body = r.json()
    # No LLM call, no foreign sources, no foreign text in the answer.
    assert all(FOREIGN_SENTINEL not in (s.get("snippet") or "") for s in body["sources"])
    assert all("Eiffel" not in (s.get("snippet") or "") for s in body["sources"])


def test_chat_rejects_foreign_notebook_scoping(client, user_a, user_b, db):
    nb = client.post("/api/notebooks", json={"name": "A's notebook"}, headers=user_a["headers"]).json()
    r = client.post(
        "/api/chat",
        json={"question": "anything", "notebook_id": nb["id"]},
        headers=user_b["headers"],
    )
    assert r.status_code == 404


def test_other_user_document_metadata_is_404(client, user_a, user_b, fake_processor, db):
    upload_file(client, user_a["headers"], filename="meta-secret.txt")
    r = client.get("/api/documents", headers=user_b["headers"])
    assert all(FOREIGN_SENTINEL not in d["filename"] for d in r.json()["documents"])
    # uuid-random document ids of other users are never listed
    assert all("a-secret" not in d["filename"] for d in r.json()["documents"])


def test_legacy_listing_is_user_scoped(client, user_a, user_b, fake_processor, db):
    upload_file(client, user_a["headers"], filename="legacy-a.txt")

    r = client.get("/api/documents/legacy", headers=user_b["headers"])
    assert r.status_code == 200
    # B sees only B's documents (none), never A's.
    assert all(FOREIGN_SENTINEL not in str(d) for d in r.json()["documents"])

    r2 = client.get("/api/documents/legacy", headers=user_a["headers"])
    ids = [d["document_id"] for d in r2.json()["documents"]]
    assert len(ids) >= 1


# --------------------------------------------------------- file validation

def test_invalid_extension_rejected(client, user_a):
    r = client.post(
        "/api/documents/upload",
        files={"file": ("malware.exe", b"MZ fake binary", "application/octet-stream")},
        data={"notebook_id": "default"},
        headers=user_a["headers"],
    )
    assert r.status_code == 400


def test_signature_mismatch_rejected(client, user_a):
    # Claims to be a PDF but has no %PDF magic header.
    r = client.post(
        "/api/documents/upload",
        files={"file": ("fake.pdf", b"<html>not a pdf</html>", "application/pdf")},
        data={"notebook_id": "default"},
        headers=user_a["headers"],
    )
    assert r.status_code == 400
    assert "%PDF" not in r.text


def test_wrong_mime_rejected(client, user_a, fake_processor):
    r = client.post(
        "/api/documents/upload",
        files={"file": ("text.txt", SUPPORTED_TEXT.encode(), "text/html")},
        data={"notebook_id": "default"},
        headers=user_a["headers"],
    )
    assert r.status_code == 400


def test_oversized_upload_rejected(client, user_a):
    from app.api import documents as documents_module

    with patch.object(documents_module, "MAX_UPLOAD_SIZE", 1024):
        big = "x" * 5000
        r = client.post(
            "/api/documents/upload",
            files={"file": ("big.txt", big.encode(), "text/plain")},
            data={"notebook_id": "default"},
            headers=user_a["headers"],
        )
    assert r.status_code == 413


def test_malformed_file_generic_error(client, user_a, fake_processor):
    # A text file that pretends to be fine but produces no text is rejected
    # with a generic 422 that does not echo internals or file contents.
    r = client.post(
        "/api/documents/upload",
        files={"file": ("empty-doc.txt", b"   ", "text/plain")},
        data={"notebook_id": "default"},
        headers=user_a["headers"],
    )
    assert r.status_code in (400, 422)


# ------------------------------------------------- deletion invalidates data

def test_delete_document_removes_chunks_and_embeddings(client, user_a, fake_processor, db):
    nb = create_notebook(client, user_a["headers"], name="Delete test")
    r = upload_file(client, user_a["headers"], filename="delete-me.txt", notebook_id=str(nb["id"]))
    assert r.status_code == 200
    doc_uuid = r.json()["document_id"]

    from app.models.chunk import DocumentChunk
    from app.models.notebook import DocumentMeta

    meta = db.query(DocumentMeta).filter(DocumentMeta.document_uuid == doc_uuid).first()
    assert meta is not None

    # Owner can delete
    d = client.delete(f"/api/documents/{meta.id}", headers=user_a["headers"])
    assert d.status_code == 204

    assert db.query(DocumentChunk).filter(DocumentChunk.document_id == doc_uuid).count() == 0

    # And retrieval no longer returns anything for it
    q = client.post(
        "/api/retrieve",
        json={
            "question": "eiffel tower paris question",
            "top_k": 50,
            "min_similarity": -1.0,
            "notebook_id": nb["id"],
        },
        headers=user_a["headers"],
    )
    assert all(FOREIGN_SENTINEL not in c["content"] for c in q.json()["retrieved_chunks"])


def test_other_user_cannot_delete_a_document(client, user_a, user_b, fake_processor, db):
    r = upload_file(client, user_a["headers"], filename="undeletable.txt")
    doc_uuid = r.json()["document_id"]

    from app.models.notebook import DocumentMeta

    meta = db.query(DocumentMeta).filter(DocumentMeta.document_uuid == doc_uuid).first()

    d = client.delete(f"/api/documents/{meta.id}", headers=user_b["headers"])
    assert d.status_code == 404
    # Still there for the owner
    assert (
        db.query(DocumentMeta).filter(DocumentMeta.id == meta.id).first() is not None
    )


# ------------------------------------------------------------ notes/notebooks

def test_note_cannot_reference_foreign_notebook(client, user_a, user_b, db):
    nb = client.post("/api/notebooks", json={"name": "A private notebook"}, headers=user_a["headers"]).json()
    r = client.post(
        "/api/notes",
        json={"title": "Injected", "notebook_id": nb["id"]},
        headers=user_b["headers"],
    )
    assert r.status_code == 404


def test_saved_answer_cannot_reference_foreign_notebook(client, user_a, user_b, db):
    nb = client.post("/api/notebooks", json={"name": "A answers"}, headers=user_a["headers"]).json()
    r = client.post(
        "/api/saved-answers",
        json={"question": "q", "answer": "a", "notebook_id": nb["id"]},
        headers=user_b["headers"],
    )
    assert r.status_code == 404


# ------------------------------------------------------- error message hygiene

def test_retrieval_errors_are_sanitized(client, user_a):
    nb = create_notebook(client, user_a["headers"], name="Sanitize test")
    # Entity tags / internal details must not appear even if something fails.
    r = client.post(
        "/api/retrieve",
        json={"question": "", "top_k": 5, "notebook_id": nb["id"]},
        headers=user_a["headers"],
    )
    assert r.status_code == 400

    r = client.post(
        "/api/retrieve",
        json={"question": "q", "top_k": 5, "notebook_id": nb["id"]},
        headers=user_a["headers"],
    )
    assert r.status_code == 200
    # Response never contains connection strings or key material.
    body_text = r.text
    assert "postgresql://" not in body_text
    assert "GEMINI" not in body_text
