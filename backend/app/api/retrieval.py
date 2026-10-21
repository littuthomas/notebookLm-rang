import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.user import User
from app.services.auth import get_current_user
from app.services.retrieval import create_retrieval_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["retrieval"])


# Request/Response models using Pydantic
class RetrievalRequest(BaseModel):
    """Request model for retrieval endpoint."""
    question: str = Field(..., description="The user's question to search for")
    top_k: int = Field(default=5, ge=1, le=100, description="Maximum number of chunks to return")
    notebook_id: int = Field(
        ...,
        ge=1,
        description=(
            "Notebook to retrieve from (mandatory). It must belong to the "
            "authenticated user; otherwise 404. Only that notebook's "
            "documents are searched."
        ),
    )


class RetrievedChunk(BaseModel):
    """Response model for a single retrieved chunk."""
    id: int
    document_id: str
    chunk_index: int
    content: str
    similarity_score: float | None = None


class RetrievalResponse(BaseModel):
    """Response model for retrieval endpoint."""
    question: str
    retrieved_chunks: list[RetrievedChunk]
    total_chunks: int


@router.post("/retrieve", response_model=RetrievalResponse)
async def retrieve_documents(
    request: RetrievalRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> RetrievalResponse:
    """
    Retrieve relevant document chunks for a given question.

    This endpoint:
    1. Takes a user question
    2. Converts it to an embedding vector
    3. Performs vector similarity search using pgvector, restricted to the
       authenticated user's own document chunks (tenant isolation)
    4. Returns the most relevant document chunks owned by the caller

    **Note:** This is the basic retrieval pipeline. Gemini answer generation
    and WebSocket chat build on top of it.

    - **question**: The user's question to search for
    - **top_k**: Maximum number of chunks to return (default: 5)
    - **notebook_id**: The notebook to retrieve from (must be owned by the caller); only that notebook's documents are searched
    """
    # --- Error Handling ---

    # 1. Validate question is not empty
    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty",
        )

    # --- Mandatory notebook scoping: the notebook must belong to the caller ---
    # Notebooks are independent knowledge spaces: retrieval is always scoped
    # to a single owned notebook, never the user's other notebooks.
    from app.models.notebook import Notebook

    notebook_owned = (
        db.query(Notebook)
        .filter(Notebook.id == request.notebook_id, Notebook.user_id == current_user.id)
        .first()
    )
    if notebook_owned is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Notebook not found",
        )

    # --- Retrieval (always scoped to the authenticated user's own chunks) ---
    try:
        retrieval_service = create_retrieval_service(db)
        chunks = retrieval_service.retrieve(
            question=request.question,
            top_k=request.top_k,
            user_id=current_user.id,
            notebook_id=request.notebook_id,
        )

        # Convert to response format
        retrieved_chunks = [
            RetrievedChunk(
                id=c["id"],
                document_id=c["document_id"],
                chunk_index=c["chunk_index"],
                content=c["content"],
                similarity_score=c.get("similarity_score"),
            )
            for c in chunks
        ]

        return RetrievalResponse(
            question=request.question,
            retrieved_chunks=retrieved_chunks,
            total_chunks=len(retrieved_chunks),
        )

    except ValueError as e:
        # Re-raise validation errors from embedding service
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    except RuntimeError as e:
        # Embedding or database failure. Raw error text (DB messages, provider
        # details) stays in the server log only — never in the response body.
        logger.error(f"Retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Retrieval is temporarily unavailable. Please try again later.",
        )


@router.get("/retrieve/sample")
async def get_sample_question() -> dict:
    """
    Get a sample question for testing the retrieval endpoint.

    Returns an example question you can use to test the retrieval pipeline.
    """
    return {
        "sample_question": "What is the capital of France?",
        "hint": "Use this question with the POST /api/retrieve endpoint",
    }
