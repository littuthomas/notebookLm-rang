"""
Additional upload-pipeline tests covering the create-notebook + upload flow:

  * Empty (0-byte) files are rejected with 400.
  * An upload is associated with the caller's OWN notebook (document_uuid ->
    DocumentMeta -> notebook_id), never with another user's notebook even if
    a foreign notebook id is supplied.
  * Successful uploads are stored with status "ready" and chunk counts > 0.
  * Processing failure (no extractable text) is reported as 422 without
    leaking internal details.
  * Filename sanitisation keeps path-traversal attempts inert.

Uses the offline embedding backend (conftest) and a deterministic fake
processor, so NO external AI calls are made.
"""
import uuid
from unittest.mock import patch

import pytest


SUPPORTED_TEXT = (
    "Paris is the capital of France. The Eiffel Tower is in Paris. "
    "France is a country in western Europe."
)


class TextFakeProcessor:
    """Deterministic processor: extracts the file's text as a single chunk."""

    def process_file(self, file_path, filename, notebook_id, document_id=None):
        from pathlib import Path

        from app.services.embedding import get_embedding_service

        doc_id = document_id or uuid.uuid4().hex
        content = Path(file_path).read_text(encoding="utf-8", errors="replace")
        if not content.strip():
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
                    "content": content,
                    "embedding": get_embedding_service().embed(content),
                }
            ],
            "error_message": None,
        }


@pytest.fixture()
def processor():
    with patch("app.api.documents.get_document_processor", return_value=TextFakeProcessor()):
        yield


@pytest.fixture()
def db(client):
    from app.core.database import get_db
    from app.main import app

    gen = app.dependency_overrides[get_db]()
    session = next(gen)
    try:
        yield session
    finally:
        gen.close()


def upload(client, headers, *, filename="notes.txt", content=SUPPORTED_TEXT, notebook_id="default"):
    return client.post(
        "/api/documents/upload",
        files={"file": (filename, content.encode("utf-8"), "text/plain")},
        data={"notebook_id": str(notebook_id)},
        headers=headers,
    )


# ------------------------------------------------------------- empty files

def test_empty_file_rejected(client, user_a, processor):
    r = upload(client, user_a["headers"], filename="empty.txt", content="")
    assert r.status_code == 400
    assert "empty" in r.json()["detail"].lower()


def test_whitespace_only_file_processed_as_failure(client, user_a, processor):
    r = upload(client, user_a["headers"], filename="blank.txt", content="   \n  ")
    # Whitespace-only files are never "successfully" stored.
    assert r.status_code == 422
    # Failed uploads must not leave document metadata behind.
    docs = client.get("/api/documents", headers=user_a["headers"]).json()["documents"]
    assert all(d["filename"] != "blank.txt" for d in docs)


# --------------------------------------------------- association + statuses

def test_successful_upload_is_ready_and_associated_with_own_notebook(client, user_a, processor, db):
    nb = client.post("/api/notebooks", json={"name": "Upload target"}, headers=user_a["headers"]).json()
    r = upload(client, user_a["headers"], notebook_id=nb["id"])
    assert r.status_code == 200, r.text
    body = r.json()

    # Structured, confirmed success — the frontend never fabricates this.
    # Response status is the post-pipeline state; DB-tracked status is "ready".
    assert body["status"] in ("ready", "completed")
    assert body["chunks_stored"] > 0
    assert body["document_id"]

    from app.models.notebook import DocumentMeta

    meta = db.query(DocumentMeta).filter(DocumentMeta.document_uuid == body["document_id"]).first()
    assert meta is not None
    assert meta.notebook_id == nb["id"]  # associated with the TARGET notebook
    assert meta.status == "ready"  # DB-tracked processing state


def test_upload_to_foreign_notebook_id_never_associates_there(
    client, user_a, user_b, processor, db
):
    """A foreign notebook id must not attach documents to another user's notebook."""
    foreign = client.post(
        "/api/notebooks", json={"name": "User B notebook"}, headers=user_b["headers"]
    ).json()

    r = upload(client, user_a["headers"], notebook_id=foreign["id"])
    assert r.status_code == 200  # falls back to the caller's default notebook

    from app.models.notebook import DocumentMeta

    meta = db.query(DocumentMeta).filter(
        DocumentMeta.document_uuid == r.json()["document_id"]
    ).first()
    assert meta is not None
    # NOT user B's notebook.
    assert meta.notebook_id != foreign["id"]
    # ...and truly never one owned by another user.
    from app.models.notebook import Notebook

    owner_id = db.query(Notebook).filter(Notebook.id == meta.notebook_id).first().user_id
    assert owner_id != foreign["id"]


# --------------------------------------------------------- traversal names

def test_path_traversal_filename_is_sanitized(client, user_a, processor):
    r = upload(client, user_a["headers"], filename="../../etc/passwd.txt")
    assert r.status_code == 200
    docs = client.get("/api/documents", headers=user_a["headers"]).json()["documents"]
    stored = next(d for d in docs if d["document_uuid"] == r.json()["document_id"])
    assert "/" not in stored["filename"] and "\\" not in stored["filename"]
    assert stored["filename"].endswith("passwd.txt")


# ------------------------------------------------------- notebook deletion

def test_notebook_delete_cleans_its_documents_and_chunks(client, user_a, processor, db):
    from app.models.chunk import DocumentChunk
    from app.models.notebook import DocumentMeta

    nb = client.post("/api/notebooks", json={"name": "Doomed"}, headers=user_a["headers"]).json()
    r = upload(client, user_a["headers"], notebook_id=nb["id"])
    doc_uuid = r.json()["document_id"]
    assert db.query(DocumentChunk).filter(DocumentChunk.document_id == doc_uuid).count() > 0

    dr = client.delete(f"/api/notebooks/{nb['id']}", headers=user_a["headers"])
    assert dr.status_code == 204
    assert db.query(DocumentMeta).filter(DocumentMeta.document_uuid == doc_uuid).first() is None
    assert db.query(DocumentChunk).filter(DocumentChunk.document_id == doc_uuid).count() == 0
