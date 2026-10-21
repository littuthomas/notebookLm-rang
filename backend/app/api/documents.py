import logging
import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import get_db
from app.models.chunk import DocumentChunk
from app.models.notebook import DocumentMeta, Notebook
from app.services.auth import get_current_user
from app.services.document_processor import get_document_processor

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/api/documents", tags=["documents"])

# Upload size limit comes from settings (env-tunable, default 20 MB).
MAX_UPLOAD_SIZE = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".xls", ".txt", ".md", ".csv"}


class UploadResponse(BaseModel):
    """Response model for the document upload endpoint."""

    document_id: str
    filename: str
    file_type: str
    status: str
    page_count: int
    chunk_count: int
    chunks_stored: int
    error_message: str | None = None


# ---------------------------------------------------------------- helpers


def sanitize_filename(filename: str) -> str:
    """Make a client-supplied filename safe to store and log.

    Strips any directory components (path traversal), control characters and
    reserves; caps the length. Only the display name is sanitized — storage
    always uses a server-generated UUID name.
    """
    # Remove any path component — "C:\\evil\\x.pdf" and "../../x.pdf" both
    # collapse to "x.pdf".
    name = Path(filename).name
    # Drop control characters and non-printable junk.
    name = "".join(ch for ch in name if ch.isprintable()).strip()
    if not name:
        name = "uploaded-document"
    return name[: settings.MAX_FILENAME_CHARS]


def validate_file_signature(contents: bytes, ext: str) -> None:
    """Validate the file's magic bytes against the declared extension.

    Rejects files whose actual content does not match the claimed type, which
    stops disguised executables/HTML/scripts from entering the pipeline.
    TXT/MD/CSV are validated as decodable text (no strict magic bytes).
    """
    if ext == ".pdf":
        if not contents.startswith(b"%PDF-"):
            raise ValueError("File content does not look like a PDF document")
    elif ext in (".docx", ".xlsx", ".xls"):
        # OOXML (.docx/.xlsx) and legacy .xls (CFB compound file) both start
        # with the OLE/zip container header "PK\x03\x04" (zip) — .xls legacy
        # actually starts with \xd0\xcf\x11\xe0.
        is_zip = contents[:4] == b"PK\x03\x04" or contents[:4] == b"PK\x05\x06"
        is_cfb = contents[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
        if not (is_zip or is_cfb):
            raise ValueError("File content does not look like a Word/Excel document")
    elif ext in (".txt", ".md", ".csv"):
        # Text files: must decode as text (robust fallback encoding at the end).
        try:
            contents[:8192].decode("utf-8")
        except UnicodeDecodeError:
            try:
                contents[:8192].decode("latin-1")
            except UnicodeDecodeError as e:  # pragma: no cover - latin-1 never fails
                raise ValueError("File content is not decodable text") from e
        # Cheap binary payload heuristic: NUL bytes are not valid text.
        if b"\x00" in contents[:8192]:
            raise ValueError("File content looks binary, not plain text")


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(
        ..., description="Document file to process (PDF, DOCX, XLSX, TXT, MD, CSV)"
    ),
    notebook_id: str = Form(default="default"),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
) -> UploadResponse:
    """
    Upload a document, process it through the pipeline, and store the chunks —
    bound to the authenticated user and their notebook.

    Supported types: PDF, Word (.docx), Excel (.xlsx), TXT, Markdown, CSV

    Pipeline:
    1. Validate extension, MIME type, size (MAX_UPLOAD_SIZE_MB, default 20 MB) and
       file signature (magic bytes)
    2. Save the bytes to a private, server-named temp file (client filename is
       never used for storage — only kept as display metadata)
    3. Extract text -> clean -> chunk -> embed (the embedding call sends each
       small chunk of TEXT ONLY to the configured provider)
    4. Persist document metadata AND chunks (with owner columns) in ONE
       database transaction — nothing is half-stored
    5. Clean up all temp files, then return a processing summary

    - **file**: The document file to upload
    - **notebook_id**: Optional notebook identifier (default: caller's default notebook)
    """
    # --- Validation ---

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required",
        )

    display_name = sanitize_filename(file.filename)
    ext = Path(display_name).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file type. Supported: PDF, DOCX, XLSX, TXT, MD, CSV",
        )

    # MIME type check (lenient: many browsers send generic types). A clearly
    # wrong declared type (e.g. text/html for a .pdf) is rejected.
    declared_mime = (file.content_type or "").split(";")[0].strip().lower()
    allowed_mimes = {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel",
        "application/msword",
        "application/excel",
        "text/plain",
        "text/markdown",
        "text/csv",
        "application/csv",
        "application/octet-stream",  # common generic fallback
    }
    if declared_mime and declared_mime not in allowed_mimes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported media type for the uploaded file",
        )

    # Read the file with a hard size cap: stop reading as soon as the limit is
    # exceeded so an oversized upload is aborted before it fills memory.
    contents = bytearray()
    chunk_buffer_size = 1024 * 1024
    while True:
        buf = await file.read(chunk_buffer_size)
        if not buf:
            break
        contents.extend(buf)
        if len(contents) > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Max size: {settings.MAX_UPLOAD_SIZE_MB} MB",
            )
    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty",
        )
    upload_size = len(contents)

    # Content/signature check BEFORE any processing or external calls.
    try:
        validate_file_signature(bytes(contents), ext)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    # Check the embedding backend is configured before doing any work
    if not settings.google_api_configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embeddings are not configured. Cannot process uploads.",
        )

    # --- Save to a private temp dir with a server-generated name ---

    temp_dir = tempfile.mkdtemp(prefix="rag_upload_")
    safe_name = f"{uuid.uuid4().hex}{ext}"
    temp_path = Path(temp_dir) / safe_name
    # Defence in depth for path traversal: the name is generated server-side,
    # but resolve and re-check containment anyway.
    if not temp_path.resolve().is_relative_to(Path(temp_dir).resolve()):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded file",
        )

    try:
        temp_path.write_bytes(bytes(contents))
    except Exception:
        logger.error("Failed to write temp file")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded file",
        )
    finally:
        await file.close()
    contents = None  # release the raw upload bytes early

    # --- Process through the pipeline ---

    processor = get_document_processor(upload_dir=temp_dir)
    result = processor.process_file(
        file_path=temp_path,
        filename=display_name,
        notebook_id=notebook_id,
    )

    # --- Handle failure ---

    if result["status"] == "failed":
        # Generic client message; server log keeps the specifics.
        logger.warning(
            "Processing failed for document %s: %s",
            result["document_id"],
            result["error_message"],
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Could not extract text from the uploaded document "
            "(unsupported or malformed file).",
        )

    # Enforce configurable page / chunk limits.
    if result["page_count"] > settings.MAX_PAGES_PER_DOC:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Document too large: exceeds the {settings.MAX_PAGES_PER_DOC} page limit",
        )
    if len(result["chunks"]) > settings.MAX_CHUNKS_PER_DOC:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Document too large: exceeds the {settings.MAX_CHUNKS_PER_DOC} chunk limit",
        )

    # --- Persist metadata AND chunks atomically (single transaction) ---

    chunks_stored = 0
    try:
        # Resolve notebook: numeric id => owned notebook, otherwise the user's
        # default notebook (created on first upload) keeps legacy behaviour.
        # Ownership is enforced by filtering on user_id — another user's
        # notebook id simply resolves to the caller's default notebook.
        notebook: Notebook | None = None
        if notebook_id.isdigit():
            notebook = (
                db.query(Notebook)
                .filter(Notebook.id == int(notebook_id), Notebook.user_id == current_user.id)
                .first()
            )
        if notebook is None:
            notebook = (
                db.query(Notebook)
                .filter(Notebook.user_id == current_user.id)
                .order_by(Notebook.created_at.asc())
                .first()
            )
        if notebook is None:
            notebook = Notebook(
                user_id=current_user.id,
                name="My Notebook",
                description="Created automatically from your first upload.",
            )
            db.add(notebook)
            db.flush()

        document_meta = DocumentMeta(
            notebook_id=notebook.id,
            document_uuid=result["document_id"],
            filename=result["filename"],
            file_type=result["file_type"],
            size_bytes=upload_size,
            page_count=result["page_count"],
            chunk_count=result["chunk_count"],
            status="ready",
        )
        db.add(document_meta)
        db.flush()

        for chunk in result["chunks"]:
            db_chunk = DocumentChunk(
                document_id=chunk["document_id"],
                chunk_index=chunk["chunk_index"],
                content=chunk["content"],
                embedding=chunk["embedding"],
                # Tenant isolation: the chunk belongs to this user and notebook.
                user_id=current_user.id,
                notebook_id=notebook.id,
            )
            db.add(db_chunk)
        chunks_stored = len(result["chunks"])
        # Everything (notebook, metadata, chunks) commits or rolls back together.
        db.commit()
        logger.info(
            "Stored %d chunks for document %s (user %s)",
            chunks_stored,
            result["document_id"],
            current_user.id,
        )
    except Exception as e:
        db.rollback()
        # Raw DB error stays in the log; the client gets a generic message.
        logger.error(f"DB persistence failed for document {result['document_id']}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Storage failure while saving the document. Please try again later.",
        )
    finally:
        # Always clean up the private temp directory (files + the dir itself).
        shutil.rmtree(temp_dir, ignore_errors=True)

    # --- Response ---

    return UploadResponse(
        document_id=result["document_id"],
        filename=result["filename"],
        file_type=result["file_type"],
        status=result["status"],
        page_count=result["page_count"],
        chunk_count=result["chunk_count"],
        chunks_stored=chunks_stored,
        error_message=None,
    )


@router.get("")
async def list_documents(
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
) -> dict:
    """
    List the current user's documents with full metadata.

    Note: the legacy chunk-count-only listing moved to
    GET /api/documents/legacy (kept for backwards compatibility).
    """
    from app.api.notebooks import list_documents_meta

    return list_documents_meta(db=db, current_user=current_user)


@router.get("/legacy")
async def list_documents_legacy(
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
) -> dict:
    """
    Legacy listing: one entry per document_id with chunk counts.
    Scoped to the authenticated user's own documents.
    """
    from sqlalchemy import func

    rows = (
        db.query(
            DocumentChunk.document_id,
            func.count(DocumentChunk.id).label("chunk_count"),
        )
        .filter(DocumentChunk.user_id == current_user.id)
        .group_by(DocumentChunk.document_id)
        .all()
    )

    documents = [
        {"document_id": row.document_id, "chunk_count": row.chunk_count}
        for row in rows
    ]

    return {"documents": documents, "total": len(documents)}
