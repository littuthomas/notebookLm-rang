"""
Tests for the documents endpoints (metadata listing, legacy listing, delete)
and the stats endpoint that aggregates document data.

Document metadata rows are created directly through the DB session because
the full upload pipeline requires the Gemini embedding API.
"""
import uuid
from datetime import datetime, timezone

import pytest


# ---------------------------------------------------------------- helpers

def make_notebook(client, headers, name=None):
    r = client.post(
        "/api/notebooks",
        json={"name": name or f"NB {uuid.uuid4().hex[:6]}"},
        headers=headers,
    )
    assert r.status_code == 201, r.text
    return r.json()


def insert_document(db_session, notebook_id, **overrides):
    """Insert a DocumentMeta row directly (bypasses the embedding pipeline)."""
    from app.models.notebook import DocumentMeta

    defaults = dict(
        notebook_id=notebook_id,
        document_uuid=uuid.uuid4().hex,
        filename="sample.pdf",
        file_type="pdf",
        size_bytes=1234,
        page_count=3,
        chunk_count=5,
        status="ready",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    defaults.update(overrides)
    doc = DocumentMeta(**defaults)
    db_session.add(doc)
    db_session.commit()
    db_session.refresh(doc)
    return doc


@pytest.fixture()
def db(client):
    """A raw DB session bound to the same session the client overrides use."""
    from app.core.database import get_db
    from app.main import app

    # Pull a session from the same overridden dependency
    gen = app.dependency_overrides[get_db]()
    session = next(gen)
    try:
        yield session
    finally:
        gen.close()


# ---------------------------------------------------------------- list (metadata)

def test_list_documents_requires_auth(client):
    r = client.get("/api/documents")
    assert r.status_code == 401


def test_list_documents_returns_full_metadata_shape(client, user_a, db):
    nb = make_notebook(client, user_a["headers"])
    insert_document(db, nb["id"])

    r = client.get("/api/documents", headers=user_a["headers"])
    assert r.status_code == 200
    docs = r.json()["documents"]
    assert len(docs) >= 1
    doc = docs[0]
    for key in (
        "id", "notebook_id", "document_uuid", "filename", "file_type",
        "size_bytes", "page_count", "chunk_count", "status", "created_at",
    ):
        assert key in doc, f"missing key: {key}"
    assert doc["file_type"] == "pdf"
    assert doc["status"] == "ready"


def test_list_documents_isolated_per_user(client, user_a, user_b, db):
    nb_a = make_notebook(client, user_a["headers"])
    nb_b = make_notebook(client, user_b["headers"])
    doc_a = insert_document(db, nb_a["id"], filename="a.pdf")
    insert_document(db, nb_b["id"], filename="b.pdf")

    r = client.get("/api/documents", headers=user_a["headers"])
    filenames = [d["filename"] for d in r.json()["documents"]]
    assert "a.pdf" in filenames
    assert "b.pdf" not in filenames
    assert doc_a is not None


# ---------------------------------------------------------------- legacy

def test_legacy_documents_listing_shape(client, user_a, db):
    nb = make_notebook(client, user_a["headers"])
    insert_document(db, nb["id"], document_uuid="legacy-uuid-123")

    # Seed matching chunks so the legacy grouping has something to count.
    # Chunks must carry the owner's user_id (in production the startup
    # tenancy migration backfills exactly this), because the legacy listing
    # is scoped to the authenticated user's own chunks.
    from app.models.chunk import DocumentChunk
    from app.models.user import User

    owner = db.query(User).filter(User.username == user_a["username"]).first()
    db.add(DocumentChunk(document_id="legacy-uuid-123", chunk_index=0, content="c0", embedding=[0.0] * 768, user_id=owner.id))
    db.add(DocumentChunk(document_id="legacy-uuid-123", chunk_index=1, content="c1", embedding=[0.0] * 768, user_id=owner.id))
    db.commit()

    r = client.get("/api/documents/legacy", headers=user_a["headers"])
    assert r.status_code == 200
    body = r.json()
    assert "documents" in body and "total" in body
    entry = next(d for d in body["documents"] if d["document_id"] == "legacy-uuid-123")
    assert entry["chunk_count"] == 2


# ---------------------------------------------------------------- detail / delete

def test_get_document_metadata(client, user_a, db):
    nb = make_notebook(client, user_a["headers"])
    doc = insert_document(db, nb["id"], filename="detail.pdf")

    r = client.get(f"/api/documents/{doc.id}", headers=user_a["headers"])
    assert r.status_code == 200
    assert r.json()["filename"] == "detail.pdf"


def test_get_document_of_other_user_404(client, user_a, user_b, db):
    nb_b = make_notebook(client, user_b["headers"])
    doc_b = insert_document(db, nb_b["id"])

    r = client.get(f"/api/documents/{doc_b.id}", headers=user_a["headers"])
    assert r.status_code == 404


def test_delete_document_removes_metadata_and_chunks(client, user_a, db):
    nb = make_notebook(client, user_a["headers"])
    doc = insert_document(db, nb["id"], document_uuid="delete-me-uuid")

    from app.models.chunk import DocumentChunk

    db.add(DocumentChunk(document_id="delete-me-uuid", chunk_index=0, content="c", embedding=[0.0] * 768))
    db.commit()

    r = client.delete(f"/api/documents/{doc.id}", headers=user_a["headers"])
    assert r.status_code == 204

    # Metadata gone
    from app.models.notebook import DocumentMeta

    assert db.query(DocumentMeta).filter(DocumentMeta.id == doc.id).first() is None
    # Chunks for that document gone
    assert (
        db.query(DocumentChunk).filter(DocumentChunk.document_id == "delete-me-uuid").first()
        is None
    )


def test_delete_document_of_other_user_404(client, user_a, user_b, db):
    nb_b = make_notebook(client, user_b["headers"])
    doc_b = insert_document(db, nb_b["id"])

    r = client.delete(f"/api/documents/{doc_b.id}", headers=user_a["headers"])
    assert r.status_code == 404
    # Still exists
    r = client.get(f"/api/documents/{doc_b.id}", headers=user_b["headers"])
    assert r.status_code == 200


# ---------------------------------------------------------------- stats

def test_stats_counts_user_data_only(client, user_a, user_b, db):
    nb_a = make_notebook(client, user_a["headers"])
    nb_b = make_notebook(client, user_b["headers"])
    insert_document(db, nb_a["id"], chunk_count=4)
    insert_document(db, nb_b["id"], chunk_count=99)

    # Expected values are derived from the DB because user_a is session-scoped
    # and already owns documents created by earlier tests in this session.
    from app.models.notebook import DocumentMeta, Notebook
    from app.models.user import User

    user = db.query(User).filter(User.username == user_a["username"]).first()
    owned = (
        db.query(DocumentMeta)
        .join(Notebook, DocumentMeta.notebook_id == Notebook.id)
        .filter(Notebook.user_id == user.id)
        .all()
    )
    expected_docs = len(owned)
    expected_chunks = sum(d.chunk_count for d in owned)

    r = client.get("/api/stats", headers=user_a["headers"])
    assert r.status_code == 200
    stats = r.json()
    assert stats["notebooks"] >= 1
    assert stats["documents"] == expected_docs  # A's own docs only — never B's
    assert stats["total_chunks"] == expected_chunks  # 99-chunk doc of B not counted


def test_stats_requires_auth(client):
    r = client.get("/api/stats")
    assert r.status_code == 401
