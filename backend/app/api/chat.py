import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.user import User
from app.schemas.chat import ChatRequest, ChatResponse, ChatSource
from app.services.auth import get_current_user
from app.services.chat_history import create_chat_history_service
from app.services.rag import create_rag_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Ask a question and get a RAG-generated answer",
)
async def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ChatResponse:
    """
    Ask a question and receive an answer grounded in the indexed documents.

    This is the full end-to-end RAG pipeline:

    1. Authenticates the caller with the existing JWT dependency
    2. Embeds the question with Gemini Embedding 2 (768 dimensions)
    3. Retrieves the Top-K chunks via pgvector cosine similarity
    4. Builds a context string from those chunks
    5. Asks Gemini to answer using **only** that context
    6. Returns the generated answer plus the sources it was grounded in

    If no chunk clears the minimum similarity threshold, the endpoint returns
    ``found_context: false`` with an explanatory answer instead of calling the
    LLM, so no answer is hallucinated.

    - **question**: The question to answer from the indexed documents
    - **top_k**: Optional override for the number of chunks to retrieve
    - **min_similarity**: Optional override for the relevance threshold
    """
    # --- Validation ---
    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty",
        )

    # --- RAG pipeline (retrieval + generation live in the service layer) ---
    try:
        rag_service = create_rag_service(db)
        result = rag_service.answer_question(
            question=request.question,
            top_k=request.top_k,
            min_similarity=request.min_similarity,
        )

        response = ChatResponse(
            question=request.question,
            answer=result["answer"],
            sources=[ChatSource(**source) for source in result["sources"]],
            found_context=result["found_context"],
            model=result["model"],
        )

    except ValueError as e:
        # Validation errors raised by the embedding/generation services
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    except RuntimeError as e:
        # Embedding, vector search, Gemini or database failure
        logger.error(f"RAG pipeline failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG pipeline failed: {str(e)}",
        )

    # --- Persist the exchange (additive; requires an owned session_id) ---
    # Runs only after a successful RAG result. A failed/foreign session_id is
    # logged and ignored so history problems never break the answer flow.
    if request.session_id is not None:
        try:
            history_service = create_chat_history_service(db)
            saved = history_service.record_exchange(
                session_id=request.session_id,
                user_id=current_user.id,
                question=request.question,
                answer=result["answer"],
            )
            if saved is None:
                logger.warning(
                    "Chat history save skipped: session %s not found or not owned "
                    "by user %s",
                    request.session_id,
                    current_user.id,
                )
            else:
                logger.info(
                    "Chat exchange stored in session %s (user %s)",
                    request.session_id,
                    current_user.id,
                )
        except Exception as e:
            # History must never break answering.
            logger.error(f"Failed to store chat exchange: {e}", exc_info=True)

    return response
