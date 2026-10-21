"""
API routes for notebooks, documents metadata, notes, saved answers and stats.

These endpoints EXTEND the existing RAG backend (auth, upload, chat are
untouched). All data endpoints require JWT auth.
"""
import json
import logging

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.notebook import DocumentMeta, Note, Notebook, SavedAnswer
from app.models.user import User
from app.services.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["notebooks"])


# --------------------------------------------------------------------- schemas

class NotebookCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    description: str = Field(default="", max_length=2000)
    color: str = Field(default="lavender", max_length=20)

    @field_validator("name")
    @classmethod
    def _reject_blank_name(cls, v: str) -> str:
        """Whitespace-only names are blank (422), same as empty strings."""
        if not v or not v.strip():
            raise ValueError("Name cannot be blank")
        return v


class NotebookUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=2000)
    color: str | None = Field(default=None, max_length=20)
    is_pinned: bool | None = None


class NoteCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(default="", max_length=50000)
    tags: list[str] = Field(default_factory=list)
    notebook_id: int | None = None
    is_pinned: bool = False


class NoteUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    content: str | None = Field(default=None, max_length=50000)
    tags: list[str] | None = None
    notebook_id: int | None = None
    is_pinned: bool | None = None


class SavedAnswerCreate(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    answer: str = Field(..., min_length=1, max_length=50000)
    sources: list[dict] = Field(default_factory=list)
    notebook_id: int | None = None


# ---------------------------------------------------------------- notebook CRUD

def _validate_notebook_reference(
    db: Session, notebook_id: int | None, user: User
) -> int | None:
    """Validate a client-supplied notebook_id for notes / saved answers.

    The id must reference a notebook owned by the current user; anything else
    (including another user's notebook) is a 404. None ("not linked") passes
    through untouched.
    """
    if notebook_id is None:
        return None
    notebook = (
        db.query(Notebook)
        .filter(Notebook.id == notebook_id, Notebook.user_id == user.id)
        .first()
    )
    if notebook is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Notebook not found")
    return notebook_id


def _get_owned_notebook(db: Session, notebook_id: int, user: User) -> Notebook:
    notebook = (
        db.query(Notebook)
        .filter(Notebook.id == notebook_id, Notebook.user_id == user.id)
        .first()
    )
    if notebook is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Notebook not found"
        )
    return notebook


def _notebook_to_dict(db: Session, notebook: Notebook) -> dict:
    doc_count = (
        db.query(func.count(DocumentMeta.id))
        .filter(DocumentMeta.notebook_id == notebook.id)
        .scalar()
    ) or 0
    chunk_count = (
        db.query(func.count(DocumentMeta.id))
        .filter(DocumentMeta.notebook_id == notebook.id)
        .scalar()
    ) or 0
    # Saved-answer / question counts
    question_count = (
        db.query(func.count(SavedAnswer.id))
        .filter(SavedAnswer.notebook_id == notebook.id)
        .scalar()
    ) or 0

    return {
        "id": notebook.id,
        "name": notebook.name,
        "description": notebook.description,
        "color": notebook.color,
        "is_pinned": notebook.is_pinned,
        "source_count": doc_count,
        "chunk_count": chunk_count,
        "question_count": question_count,
        "created_at": notebook.created_at.isoformat(),
        "updated_at": notebook.updated_at.isoformat(),
    }


@router.get("/notebooks")
def list_notebooks(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    """List the current user's notebooks with source/question counts."""
    notebooks = (
        db.query(Notebook)
        .filter(Notebook.user_id == current_user.id)
        .order_by(Notebook.is_pinned.desc(), Notebook.updated_at.desc())
        .all()
    )
    return {
        "notebooks": [_notebook_to_dict(db, n) for n in notebooks],
        "total": len(notebooks),
    }


@router.post("/notebooks", status_code=status.HTTP_201_CREATED)
def create_notebook(
    request: NotebookCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Create a new notebook."""
    notebook = Notebook(
        user_id=current_user.id,
        name=request.name.strip(),
        description=request.description.strip(),
        color=request.color,
    )
    db.add(notebook)
    db.commit()
    db.refresh(notebook)
    logger.info(f"Notebook created: {notebook.name} (user {current_user.id})")
    return _notebook_to_dict(db, notebook)


@router.get("/notebooks/{notebook_id}")
def get_notebook(
    notebook_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Get a single notebook with its documents, notes and saved answers."""
    notebook = _get_owned_notebook(db, notebook_id, current_user)
    data = _notebook_to_dict(db, notebook)

    docs = (
        db.query(DocumentMeta)
        .filter(DocumentMeta.notebook_id == notebook.id)
        .order_by(DocumentMeta.created_at.desc())
        .all()
    )
    notes = (
        db.query(Note)
        .filter(Note.notebook_id == notebook.id)
        .order_by(Note.is_pinned.desc(), Note.updated_at.desc())
        .all()
    )
    saved = (
        db.query(SavedAnswer)
        .filter(SavedAnswer.notebook_id == notebook.id)
        .order_by(SavedAnswer.created_at.desc())
        .all()
    )

    data["documents"] = [_document_to_dict(d) for d in docs]
    data["notes"] = [_note_to_dict(n) for n in notes]
    data["saved_answers"] = [_saved_to_dict(s) for s in saved]
    return data


@router.patch("/notebooks/{notebook_id}")
def update_notebook(
    notebook_id: int,
    request: NotebookUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Rename / recolor / pin a notebook."""
    notebook = _get_owned_notebook(db, notebook_id, current_user)
    if request.name is not None:
        notebook.name = request.name.strip()
    if request.description is not None:
        notebook.description = request.description.strip()
    if request.color is not None:
        notebook.color = request.color
    if request.is_pinned is not None:
        notebook.is_pinned = request.is_pinned
    db.commit()
    db.refresh(notebook)
    return _notebook_to_dict(db, notebook)


@router.delete("/notebooks/{notebook_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_notebook(
    notebook_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> None:
    """Delete a notebook, its linked metadata AND its stored chunks.

    The ORM cascade deletes DocumentMeta/Note/SavedAnswer rows, but chunks in
    document_chunks are linked by document_uuid, not FK — they must be removed
    explicitly, otherwise orphaned text + embeddings would remain retrievable
    across users.
    """
    from app.models.chunk import DocumentChunk

    notebook = _get_owned_notebook(db, notebook_id, current_user)

    # Collect the document ids first (they die with the cascade).
    doc_uuids = (
        db.query(DocumentMeta.document_uuid)
        .filter(DocumentMeta.notebook_id == notebook.id)
        .all()
    )
    uuid_values = [row[0] for row in doc_uuids]

    if uuid_values:
        # Restrict the delete to the owner: a matching document_uuid owned by
        # someone else (should not exist, UUIDs are random) is never touched.
        db.query(DocumentChunk).filter(
            DocumentChunk.document_id.in_(uuid_values),
            DocumentChunk.user_id == current_user.id,
        ).delete(synchronize_session=False)

    db.delete(notebook)
    db.commit()


# ------------------------------------------------------------- document metadata

def _document_to_dict(d: DocumentMeta) -> dict:
    return {
        "id": d.id,
        "notebook_id": d.notebook_id,
        "document_uuid": d.document_uuid,
        "filename": d.filename,
        "file_type": d.file_type,
        "size_bytes": d.size_bytes,
        "page_count": d.page_count,
        "chunk_count": d.chunk_count,
        "status": d.status,
        "error_message": d.error_message,
        "created_at": d.created_at.isoformat(),
        "updated_at": d.updated_at.isoformat(),
    }


@router.get("/documents")
def list_documents_meta(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    """List all of the user's documents with full metadata."""
    docs = (
        db.query(DocumentMeta)
        .join(Notebook, DocumentMeta.notebook_id == Notebook.id)
        .filter(Notebook.user_id == current_user.id)
        .order_by(DocumentMeta.created_at.desc())
        .all()
    )
    return {"documents": [_document_to_dict(d) for d in docs], "total": len(docs)}


@router.get("/documents/{document_id}")
def get_document_meta(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Get one document's metadata."""
    doc = (
        db.query(DocumentMeta)
        .join(Notebook, DocumentMeta.notebook_id == Notebook.id)
        .filter(DocumentMeta.id == document_id, Notebook.user_id == current_user.id)
        .first()
    )
    if doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return _document_to_dict(doc)


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> None:
    """Delete a document's metadata and its chunks."""
    from app.models.chunk import DocumentChunk

    doc = (
        db.query(DocumentMeta)
        .join(Notebook, DocumentMeta.notebook_id == Notebook.id)
        .filter(DocumentMeta.id == document_id, Notebook.user_id == current_user.id)
        .first()
    )
    if doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    # Owner-restricted chunk delete: the caller has already proven ownership
    # of the DOCUMENT (doc -> notebook -> user above), so its chunks are safe
    # to remove by document_uuid. Chunks whose user_id is still NULL (legacy
    # rows inserted before the tenancy migration ran its backfill) are removed
    # too — otherwise they would linger with no parent document and could be
    # re-attributed to another user by the startup backfill.
    db.query(DocumentChunk).filter(
        DocumentChunk.document_id == doc.document_uuid,
        (DocumentChunk.user_id == current_user.id) | (DocumentChunk.user_id.is_(None)),
    ).delete(synchronize_session=False)
    db.delete(doc)
    db.commit()


# ------------------------------------------------------------------------ notes

def _note_to_dict(n: Note) -> dict:
    return {
        "id": n.id,
        "notebook_id": n.notebook_id,
        "title": n.title,
        "content": n.content,
        "tags": [t for t in n.tags.split(",") if t],
        "is_pinned": n.is_pinned,
        "created_at": n.created_at.isoformat(),
        "updated_at": n.updated_at.isoformat(),
    }


@router.get("/notes")
def list_notes(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    """List the current user's notes."""
    notes = (
        db.query(Note)
        .filter(Note.user_id == current_user.id)
        .order_by(Note.is_pinned.desc(), Note.updated_at.desc())
        .all()
    )
    return {"notes": [_note_to_dict(n) for n in notes], "total": len(notes)}


@router.post("/notes", status_code=status.HTTP_201_CREATED)
def create_note(
    request: NoteCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Create a note."""
    # notebook_id must reference a notebook owned by the caller.
    _validate_notebook_reference(db, request.notebook_id, current_user)
    note = Note(
        user_id=current_user.id,
        notebook_id=request.notebook_id,
        title=request.title.strip(),
        content=request.content,
        tags=",".join(t.strip() for t in request.tags if t.strip()),
        is_pinned=request.is_pinned,
    )
    db.add(note)
    db.commit()
    db.refresh(note)
    return _note_to_dict(note)


@router.patch("/notes/{note_id}")
def update_note(
    note_id: int,
    request: NoteUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Update a note's title/content/tags/pin state."""
    note = (
        db.query(Note)
        .filter(Note.id == note_id, Note.user_id == current_user.id)
        .first()
    )
    if note is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Note not found")

    if request.title is not None:
        note.title = request.title.strip()
    if request.content is not None:
        note.content = request.content
    if request.tags is not None:
        note.tags = ",".join(t.strip() for t in request.tags if t.strip())
    if request.notebook_id is not None:
        # Ownership check on notebook reassignments too.
        _validate_notebook_reference(db, request.notebook_id, current_user)
        note.notebook_id = request.notebook_id
    if request.is_pinned is not None:
        note.is_pinned = request.is_pinned
    db.commit()
    db.refresh(note)
    return _note_to_dict(note)


@router.delete("/notes/{note_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_note(
    note_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> None:
    """Delete a note."""
    note = (
        db.query(Note)
        .filter(Note.id == note_id, Note.user_id == current_user.id)
        .first()
    )
    if note is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Note not found")
    db.delete(note)
    db.commit()


# ---------------------------------------------------------------- saved answers

def _saved_to_dict(s: SavedAnswer) -> dict:
    try:
        sources = json.loads(s.sources_json)
    except Exception:
        sources = []
    return {
        "id": s.id,
        "notebook_id": s.notebook_id,
        "question": s.question,
        "answer": s.answer,
        "sources": sources,
        "source_count": s.source_count,
        "created_at": s.created_at.isoformat(),
    }


@router.get("/saved-answers")
def list_saved_answers(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    """List the user's saved AI answers."""
    saved = (
        db.query(SavedAnswer)
        .filter(SavedAnswer.user_id == current_user.id)
        .order_by(SavedAnswer.created_at.desc())
        .all()
    )
    return {"saved_answers": [_saved_to_dict(s) for s in saved], "total": len(saved)}


@router.post("/saved-answers", status_code=status.HTTP_201_CREATED)
def create_saved_answer(
    request: SavedAnswerCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Save an AI answer from chat."""
    # notebook_id must reference a notebook owned by the caller.
    _validate_notebook_reference(db, request.notebook_id, current_user)
    saved = SavedAnswer(
        user_id=current_user.id,
        notebook_id=request.notebook_id,
        question=request.question.strip(),
        answer=request.answer,
        sources_json=json.dumps(request.sources),
        source_count=len(request.sources),
    )
    db.add(saved)
    db.commit()
    db.refresh(saved)
    return _saved_to_dict(saved)


@router.delete("/saved-answers/{saved_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_saved_answer(
    saved_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> None:
    """Delete a saved answer."""
    saved = (
        db.query(SavedAnswer)
        .filter(SavedAnswer.id == saved_id, SavedAnswer.user_id == current_user.id)
        .first()
    )
    if saved is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Saved answer not found")
    db.delete(saved)
    db.commit()


# ------------------------------------------------------------------------ stats

@router.get("/stats")
def get_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict:
    """Dashboard statistics for the current user."""
    notebook_count = (
        db.query(func.count(Notebook.id)).filter(Notebook.user_id == current_user.id).scalar()
    ) or 0
    doc_count = (
        db.query(func.count(DocumentMeta.id))
        .join(Notebook, DocumentMeta.notebook_id == Notebook.id)
        .filter(Notebook.user_id == current_user.id)
        .scalar()
    ) or 0
    chunk_count = (
        db.query(func.coalesce(func.sum(DocumentMeta.chunk_count), 0))
        .join(Notebook, DocumentMeta.notebook_id == Notebook.id)
        .filter(Notebook.user_id == current_user.id)
        .scalar()
    ) or 0
    saved_count = (
        db.query(func.count(SavedAnswer.id)).filter(SavedAnswer.user_id == current_user.id).scalar()
    ) or 0
    note_count = (
        db.query(func.count(Note.id)).filter(Note.user_id == current_user.id).scalar()
    ) or 0

    return {
        "notebooks": notebook_count,
        "documents": doc_count,
        "questions": saved_count,
        "ai_insights": note_count,
        "total_chunks": chunk_count,
    }
