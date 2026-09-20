"""Service for chat sessions and message history.

All session reads/writes go through this service so ownership enforcement
and title derivation live in exactly one place. The REST endpoints (Stage 2)
and the WebSocket flow (Stage 3) both call these methods.
"""

import logging

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.chat import ChatMessage, ChatSession
from app.models.user import User

logger = logging.getLogger(__name__)

# Title assigned to a session until the first question names it.
DEFAULT_SESSION_TITLE = "New chat"

# Max characters for a derived session title (title column is String(200)).
TITLE_MAX_CHARS = 60


def derive_title(question: str) -> str:
    """Derive a session title from the user's first question.

    Takes the first ~60 characters of the question, trimmed at a word
    boundary, with an ellipsis appended when truncated.
    """
    text = " ".join((question or "").split())
    if not text:
        return DEFAULT_SESSION_TITLE

    if len(text) <= TITLE_MAX_CHARS:
        return text

    cut = text[:TITLE_MAX_CHARS]
    # Trim back to the last full word so the title never ends mid-word.
    trimmed = cut.rsplit(" ", 1)[0]
    if len(trimmed) < 20:
        # Word boundary too far back (very long first word); keep the cut.
        trimmed = cut
    return trimmed.rstrip(" ,.;:!?\t") + "…"


class ChatHistoryService:
    """Chat session and message persistence, scoped to the owning user."""

    def __init__(self, db: Session) -> None:
        self.db = db

    # ------------------------------------------------------------------
    # Ownership gate (single source of truth for access control)
    # ------------------------------------------------------------------

    def get_owned_session(self, session_id: int, user_id: int) -> ChatSession | None:
        """Return the session only if it exists AND belongs to ``user_id``.

        Returns None when the session is missing or owned by someone else,
        so callers raise 404 without revealing other users' session ids.
        """
        return (
            self.db.query(ChatSession)
            .filter(ChatSession.id == session_id, ChatSession.user_id == user_id)
            .first()
        )

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def create_session(self, user: User) -> ChatSession:
        """Create a new empty session for ``user``."""
        session = ChatSession(user_id=user.id, title=DEFAULT_SESSION_TITLE)
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)
        logger.info(f"Chat session created: id={session.id} user={user.username}")
        return session

    def list_sessions(self, user_id: int) -> list[ChatSession]:
        """List the user's sessions, newest activity first.

        Each returned session has ``message_count`` set (without loading
        the messages themselves).
        """
        message_count = (
            select(func.count())
            .select_from(ChatMessage)
            .where(ChatMessage.session_id == ChatSession.id)
            .scalar_subquery()
        )
        return (
            self.db.query(ChatSession)
            .filter(ChatSession.user_id == user_id)
            .add_columns(message_count.label("message_count"))
            .order_by(ChatSession.updated_at.desc(), ChatSession.id.desc())
            .all()
        )

    def get_session_with_messages(self, session_id: int, user_id: int) -> ChatSession | None:
        """Return the user's session with messages ordered by created_at."""
        session = self.get_owned_session(session_id, user_id)
        if session is None:
            return None
        # Touch the ordered relationship so messages load in created_at order.
        _ = session.messages
        return session

    def delete_session(self, session_id: int, user_id: int) -> tuple[ChatSession, int] | None:
        """Delete the user's session and its messages (ORM cascade).

        Returns (session, messages_deleted) or None when not found/owned.
        """
        session = self.get_owned_session(session_id, user_id)
        if session is None:
            return None
        messages_deleted = len(session.messages)
        self.db.delete(session)
        self.db.commit()
        logger.info(
            f"Chat session deleted: id={session_id} messages={messages_deleted}"
        )
        return session, messages_deleted

    # ------------------------------------------------------------------
    # Message persistence (shared by REST now, WebSocket in Stage 3)
    # ------------------------------------------------------------------

    def record_exchange(
        self,
        session_id: int,
        user_id: int,
        question: str,
        answer: str,
    ) -> tuple[ChatMessage, ChatMessage] | None:
        """Store one user question and its assistant answer in a session.

        Ownership is re-verified before any write. On the first exchange of
        a default-titled session, the title is derived from the question.

        Returns (user_message, assistant_message), or None when the session
        does not exist or belongs to another user.
        """
        session = self.get_owned_session(session_id, user_id)
        if session is None:
            return None

        user_message = ChatMessage(role="user", content=question)
        assistant_message = ChatMessage(role="assistant", content=answer)
        session.messages.extend([user_message, assistant_message])

        if session.title == DEFAULT_SESSION_TITLE:
            session.title = derive_title(question)

        # Touch updated_at explicitly: SQLAlchemy's onupdate only fires when
        # the parent ROW changes, and adding child messages does not dirty it.
        from datetime import datetime, timezone

        session.updated_at = datetime.now(timezone.utc)
        self.db.commit()
        self.db.refresh(user_message)
        self.db.refresh(assistant_message)
        return user_message, assistant_message


def create_chat_history_service(db: Session) -> ChatHistoryService:
    """Factory matching the existing service conventions."""
    return ChatHistoryService(db)
