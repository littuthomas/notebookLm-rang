import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.database import get_db
from app.models.user import User
from app.schemas.ws import WSStartMessage
from app.services.auth import decode_access_token
from app.services.chat_history import create_chat_history_service
from app.services.ws_rag import WSRAGOrchestrator

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
                orchestrator = WSRAGOrchestrator(db=db_session)

                # Resolve the chat session (Stage 3): only an owned session is
                # used for persistence; anything else degrades to "no history".
                history_session_id: int | None = None
                if start.session_id is not None:
                    try:
                        user = (
                            db_session.query(User)
                            .filter(User.username == username)
                            .first()
                        )
                        if user is not None and (
                            create_chat_history_service(db_session).get_owned_session(
                                start.session_id, user.id
                            )
                            is not None
                        ):
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
                            user = (
                                db_session.query(User)
                                .filter(User.username == username)
                                .first()
                            )
                            if user is not None:
                                saved = create_chat_history_service(
                                    db_session
                                ).record_exchange(
                                    session_id=history_session_id,
                                    user_id=user.id,
                                    question=start.question,
                                    answer="".join(answer_parts),
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
