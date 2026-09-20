import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.user import User
from app.schemas.chat_history import (
    DeleteSessionResponse,
    MessageOut,
    SessionDetail,
    SessionListResponse,
    SessionSummary,
)
from app.services.auth import get_current_user
from app.services.chat_history import create_chat_history_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat-history"])


@router.post(
    "/sessions",
    response_model=SessionSummary,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new chat session for the logged-in user",
)
def create_session(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SessionSummary:
    """Create a new, empty chat session owned by the authenticated user."""
    service = create_chat_history_service(db)
    session = service.create_session(current_user)
    return SessionSummary(
        id=session.id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
        message_count=0,
    )


@router.get(
    "/sessions",
    response_model=SessionListResponse,
    summary="List the logged-in user's chat sessions",
)
def list_sessions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SessionListResponse:
    """List only the caller's sessions, newest activity first."""
    service = create_chat_history_service(db)
    rows = service.list_sessions(current_user.id)
    sessions = [
        SessionSummary(
            id=session.id,
            title=session.title,
            created_at=session.created_at,
            updated_at=session.updated_at,
            message_count=count,
        )
        for session, count in rows
    ]
    return SessionListResponse(sessions=sessions, total=len(sessions))


@router.get(
    "/sessions/{session_id}",
    response_model=SessionDetail,
    summary="Get one of the logged-in user's chat sessions with its messages",
)
def get_session(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> SessionDetail:
    """Return the caller's session with messages ordered by creation time.

    Returns 404 when the session does not exist **or** belongs to another
    user (ownership is never revealed through the status code).
    """
    service = create_chat_history_service(db)
    session = service.get_session_with_messages(session_id, current_user.id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session not found",
        )
    return SessionDetail(
        id=session.id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
        messages=[MessageOut.model_validate(m) for m in session.messages],
    )


@router.delete(
    "/sessions/{session_id}",
    response_model=DeleteSessionResponse,
    summary="Delete one of the logged-in user's chat sessions",
)
def delete_session(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> DeleteSessionResponse:
    """Delete the caller's session; its messages are removed by cascade.

    Returns 404 when the session does not exist or belongs to another user.
    """
    service = create_chat_history_service(db)
    result = service.delete_session(session_id, current_user.id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat session not found",
        )
    session, messages_deleted = result
    return DeleteSessionResponse(
        deleted=True,
        session_id=session.id,
        messages_deleted=messages_deleted,
    )
