import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.database import get_db
from app.models.user import User
from app.schemas.ws import WSStartMessage
from app.services.auth import decode_access_token
from app.services.chat_history import create_chat_history_service
from app.services.ws_rag import WSRAGOrchestrator

# Only set at import time so type-checkers and docs can see the name; the real
# value is injected per-request by the test fixture below. In production this is
# None and the handler builds the real retrieval service itself.
# mypy: disable-error-code="var-annotated-with-implicit-any-type"
_stub_retrieval = None

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


@router.websocket("/api/chat/ws")
async def chat_ws(websocket: WebSocket) -> None:
    """WebSocket endpoint for streaming RAG question-answering.

    Flow:
        1. Client connects.
        2. Client sends a ``start`` message with a JWT token + question.
        3. Server authenticates using the existing JWT decode logic.
        4. Server retrieves relevant chunks (reusing RetrievalService).
        5. If no context is found, server sends a clear completion and closes.
        6. Otherwise server streams Gemini answer chunks progressively.
        7. Server sends a completion message and keeps the socket open until
           the client disconnects.

    Authentication:
        The JWT access token is expected in the ``token`` field of the first
        ``start`` message. This is the simplest approach for a beginner project
        and avoids coupling WebSocket connect negotiation to the HTTP-only
        OAuth2PasswordBearer dependency.

    Chat history (optional):
        When the ``start`` message carries a ``session_id`` owned by the
        authenticated user, the question and the fully streamed answer are
        stored in that session after the ``complete`` message is sent. A
        missing/foreign ``session_id`` never breaks streaming: a warning is
        logged and the answer is delivered exactly as before. Without
        ``session_id`` the protocol behaves exactly as before.

    The existing POST /api/chat endpoint is intentionally untouched.
    """
    await websocket.accept()
    logger.info("WebSocket connection accepted")

    db_session = None
    try:
        async for message in websocket.iter_json():
            if message.get("type") == "start":
                try:
                    start = WSStartMessage(**message)
                except Exception as e:
                    await websocket.send_json(
                        {"type": "error", "detail": f"Invalid start message: {e}"}
                    )
                    break

                # Authenticate BEFORE acquiring a DB session. Also resolve
                # the identity (sub = username) to the User row so history
                # persistence can verify session ownership.
                try:
                    payload = decode_access_token(start.token)
                    username: str | None = payload.get("sub")
                except Exception as e:
                    await websocket.send_json(
                        {"type": "error", "detail": f"Authentication failed: {e}"}
                    )
                    break

                db_session = next(get_db())
                retrieval_service = _stub_retrieval if _stub_retrieval is not None else None
                orchestrator = WSRAGOrchestrator(
                    db=db_session,
                    retrieval_service=retrieval_service,
                )
                logger.debug(
                    "WS session using retrieval_service=%s",
                    type(retrieval_service).__name__,
                )

                # Resolve the authenticated user ONCE — the WS RAG flow MUST
                # scope retrieval to this user's own chunks (tenant isolation).
                # DB round-trips are remote (Neon Postgres); run every
                # synchronous query on a worker thread so the event loop is
                # never blocked while a connection round-trip is in flight.
                user = await asyncio.to_thread(
                    lambda: db_session.query(User)
                    .filter(User.username == username)
                    .first()
                )
                user_id: int | None = user.id if user is not None else None
                if user_id is None or not user.is_active:
                    await websocket.send_json(
                        {"type": "error", "detail": "Authentication failed: user not found"}
                    )
                    break

                # Mandatory notebook scoping: notebooks are independent
                # knowledge spaces, so the notebook must exist and belong to
                # the authenticated user. Without it there is no safe scope.
                from app.models.notebook import Notebook

                notebook_owned = await asyncio.to_thread(
                    lambda: db_session.query(Notebook)
                    .filter(
                        Notebook.id == start.notebook_id,
                        Notebook.user_id == user_id,
                    )
                    .first()
                )
                if notebook_owned is None:
                    await websocket.send_json(
                        {"type": "error", "detail": "Notebook not found"}
                    )
                    break
                scoped_notebook_id: int = start.notebook_id

                # Resolve the chat session (Stage 3): only an owned session is
                # used for persistence; anything else degrades to "no history".
                history_session_id: int | None = None
                if start.session_id is not None:
                    try:
                        owned_session = await asyncio.to_thread(
                            lambda: create_chat_history_service(
                                db_session
                            ).get_owned_session(start.session_id, user_id)
                        )
                        if owned_session is not None:
                            history_session_id = start.session_id
                        else:
                            logger.warning(
                                "WS history save skipped: session %s not found or "
                                "not owned by user %s",
                                start.session_id,
                                username,
                            )
                    except Exception as e:
                        logger.error(
                            f"WS history session check failed: {e}", exc_info=True
                        )

                try:
                    sent_any = False
                    answer_parts: list[str] = []
                    response: dict | None = None
                    async for response in orchestrator.handle_question(
                        token=start.token,
                        question=start.question,
                        top_k=start.top_k,
                        min_similarity=start.min_similarity,
                        user_id=user_id,
                        notebook_id=scoped_notebook_id,
                    ):
                        sent_any = True
                        # Accumulate streamed answer text server-side so the
                        # complete answer can be stored once, after "complete".
                        if isinstance(response, dict) and response.get("type") == "answer_chunk":
                            answer_parts.append(response.get("chunk") or "")
                        try:
                            await websocket.send_json(response)
                        except Exception:
                            # Client likely disconnected
                            break
                    if not sent_any:
                        logger.debug("Orchestrator produced no messages; ensuring completion")

                    # --- Persist history AFTER complete (additive) ---
                    # Only a fully streamed, successfully completed answer is
                    # stored; partial chunks are never saved. Persistence
                    # failures are logged, never surfaced as RAG errors.
                    if (
                        history_session_id is not None
                        and answer_parts
                        and isinstance(response, dict)
                        and response.get("type") == "complete"
                    ):
                        try:
                            if user is not None:
                                saved = await asyncio.to_thread(
                                    lambda: create_chat_history_service(
                                        db_session
                                    ).record_exchange(
                                        session_id=history_session_id,
                                        user_id=user.id,
                                        question=start.question,
                                        answer="".join(answer_parts),
                                    )
                                )
                                if saved is None:
                                    logger.warning(
                                        "WS history save skipped: session %s not "
                                        "found or not owned by user %s",
                                        history_session_id,
                                        username,
                                    )
                                else:
                                    logger.info(
                                        "WS chat exchange stored in session %s "
                                        "(user %s)",
                                        history_session_id,
                                        username,
                                    )
                        except Exception as e:
                            logger.error(
                                f"WS history persistence failed: {e}", exc_info=True
                            )
                except Exception as e:
                    logger.error(f"Orchestrator error: {e}", exc_info=True)
                    try:
                        await websocket.send_json(
                            {"type": "error", "detail": "Failed to process question"}
                        )
                    except Exception:
                        pass

                # After the first start, ignore further messages to keep the
                # protocol simple for a beginner project.
                break
            else:
                await websocket.send_json(
                    {"type": "error", "detail": "Expected a 'start' message first"}
                )
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected by client")
    except Exception as e:
        logger.error(f"WebSocket handler error: {e}", exc_info=True)
        try:
            await websocket.send_json({"type": "error", "detail": "Internal server error"})
        except Exception:
            pass
    finally:
        if db_session is not None:
            db_session.close()
        try:
            await websocket.close()
        except Exception:
            pass
