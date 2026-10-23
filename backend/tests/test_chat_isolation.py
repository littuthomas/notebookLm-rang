"""
Chat isolation tests for the strict per-notebook RAG pipeline.

Verified requirements:
  * Every chat/retrieval request is scoped to ONE owned notebook (mandatory).
  * Missing or foreign notebook ids are rejected before any retrieval runs.
  * Retrieved chunks, the LLM context, the answer and the citations never
    contain documents from another notebook or another user.
  * Switching notebooks per request fully isolates retrieval (no memory of the
    previous notebook's documents).
  * Unrelated questions produce an honest "not found" response without calling
    the LLM (no fabricated answers).
  * Citations reference the real retrieved chunks of the selected notebook.

NO external AI calls: embeddings use the offline simple-hash backend (conftest)
and generation is replaced with a deterministic stub, while exercising the real
routers, DB queries and pgvector search.
"""
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from app.services.generation import NO_ANSWER_MESSAGE


# ------------------------------------------------------------------ fixtures

EIFFEL_TEXT = (
    "Paris is the capital of France. The Eiffel Tower is in Paris. "
    "France is a country in western Europe."
)

OTHER_TOPIC_TEXT = (
    "Photosynthesis is how plants convert sunlight into chemical energy. "
    "Chlorophyll in leaves absorbs light during this process."
)

FOREIGN_NOTEBOOK_SENTINEL = "OTHER-NOTEBOOK-CONTENT-SENTINEL-a19f4"


class TextCaptureProcessor:
    """Deterministic document processor reading the uploaded file's text.

    Produces real chunk dicts whose embeddings come from the (offline)
    simple-hash EmbeddingService, so pgvector search behaves realistically.
    """

    def process_file(self, file_path, filename, notebook_id, document_id=None):
        from app.services.embedding import get_embedding_service

        doc_id = document_id or uuid.uuid4().hex
        content = Path(file_path).read_text(encoding="utf-8")
        embedding = get_embedding_service().embed(content)

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
                    "notebook_id": notebook_id,
                    "embedding": embedding,
                }
            ],
            "error_message": None,
        }


class StubGeneration:
    """Offline generation stub capturing what the RAG pipeline feeds the LLM.

    The stub records every (question, context) call so tests can assert that
    ONLY the selected notebook's content reaches the model, and that the
    "not found" path never invokes generation at all.
    """

    def __init__(self):
        self.calls: list[tuple[str, str]] = []
        self._model_name = "stub-grounded-model"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def last_model_used(self) -> str:
        return self._model_name

    def generate_answer(self, question: str, context: str) -> str:
        self.calls.append((question, context))
        # Grounded-style answer: the model may only echo notebook content.
        return "Grounded answer from the retrieved context. [Source 1]"


class StubRetrievalEmpty:
    """Retrieval stub that always returns no relevant chunks."""

    def retrieve(self, question, top_k=5, min_similarity=0.0, user_id=None, notebook_id=None):
        return []


@pytest.fixture()
def stub_generation():
    stub = StubGeneration()
    with patch("app.services.rag.get_generation_service", return_value=stub):
        yield stub


# ------------------------------------------------------------------- helpers


def create_notebook(client, headers, *, name="Isolation test notebook") -> dict:
    """Create an owned notebook and return its API representation."""
    r = client.post("/api/notebooks", json={"name": name}, headers=headers)
    assert r.status_code == 201 or r.status_code == 200, r.text
    return r.json()


def upload_file(client, headers, *, filename="isolation-notes.txt", content=EIFFEL_TEXT, notebook_id="default"):
    return client.post(
        "/api/documents/upload",
        files={"file": (filename, content.encode("utf-8"), "text/plain")},
        data={"notebook_id": notebook_id},
        headers=headers,
    )


def make_user_content(client, headers, *, name_prefix, content=EIFFEL_TEXT, db=None):
    """Create a notebook and upload one document into it for the given user.

    If ``db`` is provided, insert the document metadata directly so the test
    does not depend on the upload pipeline / processor patching.  Otherwise fall
    back to the normal upload path.

    Returns (notebook, document_uuid).
    """
    nb = create_notebook(client, headers, name=f"{name_prefix} notebook")
    doc_uuid = uuid.uuid4().hex

    if db is not None:
        from app.models.notebook import DocumentMeta, Notebook, User as UserModel
        from app.models.chunk import DocumentChunk

        owner_row = (
            db.query(UserModel)
            .join(Notebook, Notebook.user_id == UserModel.id)
            .filter(Notebook.id == nb["id"])
            .first()
        )
        owner_id = owner_row.id if owner_row is not None else None

        meta = DocumentMeta(
            notebook_id=nb["id"],
            document_uuid=doc_uuid,
            filename=f"{name_prefix}.txt",
            file_type="txt",
            size_bytes=len(content.encode("utf-8")),
            page_count=1,
            chunk_count=1,
            status="ready",
            error_message=None,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
        db.add(meta)
        db.flush()

        if owner_id is not None:
            chunk = DocumentChunk(
                document_id=doc_uuid,
                chunk_index=0,
                content=content,
                embedding=[0.0] * 768,
                user_id=owner_id,
                notebook_id=nb["id"],
            )
            db.add(chunk)
        db.commit()
    else:
        r = upload_file(client, headers, filename=f"{name_prefix}.txt", content=content, notebook_id=str(nb["id"]))
        assert r.status_code == 200, r.text
        doc_uuid = r.json()["document_id"]

    return nb, doc_uuid
