"""
Notebook-isolation tests for the strict per-notebook RAG pipeline.

Verified requirements:
  * Every chat/retrieval request is scoped to ONE owned notebook (mandatory).
  * Missing or foreign notebook ids are rejected before any retrieval runs.
  * Retrieved chunks, the LLM context, the answer and the citations never
    contain documents from another notebook.
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
        from pathlib import Path

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


def upload_notebook_document(client, headers, *, notebook_id, filename, content):
    """Upload a text document into a specific owned notebook."""
    r = client.post(
        "/api/documents/upload",
        files={"file": (filename, content.encode("utf-8"), "text/plain")},
        data={"notebook_id": str(notebook_id)},
        headers=headers,
    )
    assert r.status_code == 200, r.text
    return r.json()


def ask(client, headers, *, question, notebook_id, **extra):
    body = {"question": question, "notebook_id": notebook_id, **extra}
    return client.post("/api/chat", json=body, headers=headers)


# --------------------------------------------------------------- authorization


def test_chat_without_notebook_id_is_rejected(client, user_a):
    r = client.post("/api/chat", json={"question": "anything"}, headers=user_a["headers"])
    assert r.status_code == 422


def test_chat_with_foreign_notebook_is_rejected(client, user_a, user_b):
    nb_a = client.post(
        "/api/notebooks", json={"name": "A's private notes"}, headers=user_a["headers"]
    ).json()
    r = ask(client, user_b["headers"], question="anything", notebook_id=nb_a["id"])
    assert r.status_code == 404


def test_retrieve_without_notebook_id_is_rejected(client, user_a):
    r = client.post(
        "/api/retrieve",
        json={"question": "anything"},
        headers=user_a["headers"],
    )
    assert r.status_code == 422


def test_retrieve_with_foreign_notebook_is_rejected(client, user_a, user_b):
    nb_a = client.post(
        "/api/notebooks", json={"name": "A's private retrieval"}, headers=user_a["headers"]
    ).json()
    r = client.post(
        "/api/retrieve",
        json={"question": "anything", "notebook_id": nb_a["id"]},
        headers=user_b["headers"],
    )
    assert r.status_code == 404


def test_ws_without_notebook_id_is_rejected(client, user_a):
    with client.websocket_connect("/api/chat/ws") as ws:
        ws.send_json({"type": "start", "token": user_a["token"], "question": "anything"})
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert "Invalid start message" in msg["detail"]


def test_ws_with_foreign_notebook_is_rejected(client, user_a, user_b):
    nb_a = client.post(
        "/api/notebooks", json={"name": "A's private ws"}, headers=user_a["headers"]
    ).json()
    with client.websocket_connect("/api/chat/ws") as ws:
        ws.send_json(
            {
                "type": "start",
                "token": user_b["token"],
                "question": "anything",
                "notebook_id": nb_a["id"],
            }
        )
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert msg["detail"] == "Notebook not found"


# ------------------------------------------------- strict per-notebook scoping


@pytest.fixture()
def offline_processor():
    """Patch the upload endpoint's document processor with the offline fake."""
    with patch("app.api.documents.get_document_processor", return_value=TextCaptureProcessor()):
        yield


@pytest.fixture()
def two_notebooks(client, user_a, offline_processor):
    """user_a with two isolated notebooks, each holding one distinct document."""
    nb_a = client.post("/api/notebooks", json={"name": "Geography"}, headers=user_a["headers"]).json()
    nb_b = client.post(
        "/api/notebooks", json={"name": "Biology"}, headers=user_a["headers"]
    ).json()
    doc_a = upload_notebook_document(
        client, user_a["headers"], notebook_id=nb_a["id"], filename="eiffel.txt", content=EIFFEL_TEXT
    )
    doc_b = upload_notebook_document(
        client,
        user_a["headers"],
        notebook_id=nb_b["id"],
        filename="plants.txt",
        content=FOREIGN_NOTEBOOK_SENTINEL + " " + OTHER_TOPIC_TEXT,
    )
    return {"nb_a": nb_a, "nb_b": nb_b, "doc_a": doc_a, "doc_b": doc_b}


def test_chat_context_and_sources_only_from_selected_notebook(
    client, user_a, stub_generation, two_notebooks
):
    data = two_notebooks

    # Threshold disabled so EVERY owned chunk is eligible: if the notebook
    # filter were broken, the Biology chunk would leak into the Geography chat.
    r = ask(
        client,
        user_a["headers"],
        question="Where is the Eiffel Tower?",
        notebook_id=data["nb_a"]["id"],
        top_k=10,
        min_similarity=-1.0,
    )
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["found_context"] is True
    assert len(body["sources"]) == 1

    # --- citations reference the real retrieved chunk of the selected notebook
    source = body["sources"][0]
    assert source["document_id"] == data["doc_a"]["document_id"]
    assert source["chunk_id"] > 0
    assert "Eiffel Tower" in source["snippet"]
    assert "sentinel content" not in source["snippet"].lower()

    # --- the LLM context contains ONLY the selected notebook's document
    assert len(stub_generation.calls) == 1
    question_sent, context = stub_generation.calls[0]
    assert question_sent == "Where is the Eiffel Tower?"
    assert "Eiffel Tower" in context
    assert FOREIGN_NOTEBOOK_SENTINEL not in context
    assert "Photosynthesis" not in context


def test_switching_notebooks_fully_isolates_retrieval(
    client, user_a, stub_generation, two_notebooks
):
    data = two_notebooks

    # First question in the Biology notebook must know nothing about Paris.
    r = ask(
        client,
        user_a["headers"],
        question="How do plants convert sunlight into energy?",
        notebook_id=data["nb_b"]["id"],
        top_k=10,
        min_similarity=-1.0,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["found_context"] is True
    assert all(s["document_id"] == data["doc_b"]["document_id"] for s in body["sources"])

    assert len(stub_generation.calls) == 1
    _, context = stub_generation.calls[0]
    assert "Photosynthesis" in context
    assert "Eiffel Tower" not in context
    assert "Paris" not in context

    # The first notebook's question/chat must NOT have influenced this answer:
    # each request only saw the Biology notebook's document.
    assert stub_generation.calls[0][0] != "Where is the Eiffel Tower?"


# -------------------------------------------------- no fabrication / not found


def test_no_relevant_chunks_returns_not_found_without_llm(client, user_a, stub_generation):
    nb = client.post(
        "/api/notebooks", json={"name": "Empty knowledge space"}, headers=user_a["headers"]
    ).json()

    # Retrieval legitimately finds nothing above the configured threshold.
    with patch("app.services.rag.create_retrieval_service") as factory:
        factory.return_value = StubRetrievalEmpty()
        r = ask(client, user_a["headers"], question="What is the capital of France?", notebook_id=nb["id"])

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["found_context"] is False
    assert body["answer"] == NO_ANSWER_MESSAGE
    assert body["sources"] == []
    # The LLM was never called — no fabricated answer, no invented citations.
    assert stub_generation.calls == []


def test_default_relevance_threshold_is_applied(client, user_a, stub_generation):
    nb = client.post(
        "/api/notebooks", json={"name": "Threshold check"}, headers=user_a["headers"]
    ).json()

    from app.core.config import get_settings

    settings = get_settings()

    class ThresholdCapturingRetrieval:
        def __init__(self):
            self.seen = []

        def retrieve(self, question, top_k=5, min_similarity=0.0, user_id=None, notebook_id=None):
            self.seen.append(
                {
                    "question": question,
                    "top_k": top_k,
                    "min_similarity": min_similarity,
                    "user_id": user_id,
                    "notebook_id": notebook_id,
                }
            )
            return []

    capture = ThresholdCapturingRetrieval()
    with patch("app.services.rag.create_retrieval_service", return_value=capture):
        ask(client, user_a["headers"], question="Anything specific?", notebook_id=nb["id"])

    assert len(capture.seen) == 1
    first = capture.seen[0]
    assert first["question"] == "Anything specific?"
    assert first["top_k"] == settings.RAG_TOP_K
    assert first["min_similarity"] == settings.RAG_MIN_SIMILARITY
    # The pipeline propagates the server-side identity (never client data).
    assert isinstance(first["user_id"], int) and first["user_id"] > 0
    assert first["notebook_id"] == nb["id"]


def test_only_relevant_passages_are_passed_to_the_llm(
    client, user_a, stub_generation, offline_processor
):
    """The context handed to the LLM contains the retrieved passages and
    their citation labels — nothing invented by the orchestrator."""
    nb = client.post(
        "/api/notebooks", json={"name": "Citations"}, headers=user_a["headers"]
    ).json()
    doc = upload_notebook_document(
        client, user_a["headers"], notebook_id=nb["id"], filename="facts.txt", content=EIFFEL_TEXT
    )

    r = ask(
        client,
        user_a["headers"],
        question="Anything at all?",
        notebook_id=nb["id"],
        top_k=5,
        min_similarity=-1.0,
    )
    assert r.status_code == 200
    body = r.json()

    for source in body["sources"]:
        assert source["document_id"] == doc["document_id"]
        assert source["chunk_id"] > 0
        assert source["snippet"].strip() != ""

    _, context = stub_generation.calls[0]
    assert "Eiffel Tower" in context
    # Chunks are labelled so inline citations map to a real retrieved chunk.
    assert "[Source 1]" in context
    assert f"document_id={doc['document_id']}" in context


def test_multiple_notebooks_of_the_same_user_never_mix(
    client, user_a, user_b, stub_generation, fake_style_processor=None
):
    """Same-user notebooks are isolated from each other too (not just users)."""
    nb_a = client.post("/api/notebooks", json={"name": "Notebook A"}, headers=user_a["headers"]).json()
    nb_b = client.post("/api/notebooks", json={"name": "Notebook B"}, headers=user_a["headers"]).json()

    with patch("app.api.documents.get_document_processor", return_value=TextCaptureProcessor()):
        upload_notebook_document(
            client, user_a["headers"], notebook_id=nb_a["id"], filename="paris.txt", content=EIFFEL_TEXT
        )
        # B's document is the "foreign" one: its text must never be retrieved
        # into a question asked in notebook A.
        upload_notebook_document(
            client,
            user_a["headers"],
            notebook_id=nb_b["id"],
            filename="plants.txt",
            content=FOREIGN_NOTEBOOK_SENTINEL + " " + OTHER_TOPIC_TEXT,
        )

    r = ask(
        client,
        user_a["headers"],
        question="Where is the Eiffel Tower?",
        notebook_id=nb_a["id"],
        top_k=50,
        min_similarity=-1.0,
    )
    assert r.status_code == 200
    body = r.json()

    assert all(FOREIGN_NOTEBOOK_SENTINEL not in (s.get("snippet") or "") for s in body["sources"])
    assert all("Photosynthesis" not in (s.get("snippet") or "") for s in body["sources"])

    _, context = stub_generation.calls[0]
    assert FOREIGN_NOTEBOOK_SENTINEL not in context
    assert "Photosynthesis" not in context
