import logging
from typing import AsyncIterator

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.schemas.ws import (
    WSAnswerChunkMessage,
    WSCompleteMessage,
    WSContextMessage,
    WSErrorMessage,
)
from app.services.auth import decode_access_token
from app.services.generation import (
    GenerationService,
    NO_ANSWER_MESSAGE,
    get_generation_service,
)
from app.services.retrieval import RetrievalService, create_retrieval_service

logger = logging.getLogger(__name__)
settings = get_settings()

# Set DEBUG_WS_FLOW=1 in .env to log exactly which WS messages are yielded.
DEBUG_WS_FLOW = getattr(settings, "DEBUG_WS_FLOW", False)


class WSRAGOrchestrator:
    """Orchestrates the streaming RAG flow for a single WebSocket session.

    This class is intentionally separate from the WebSocket route so the route
    file stays small and the RAG logic can be tested/reused independently.
    It reuses the existing RetrievalService and GenerationService rather than
    duplicating the vector-search or Gemini setup.
    """

    def __init__(self, db: Session, generation_service: GenerationService | None = None) -> None:
        self.db = db
        self.retrieval_service = create_retrieval_service(db)
        self.generation_service = generation_service or get_generation_service()

    async def handle_question(
        self,
        token: str,
        question: str,
        top_k: int | None = None,
        min_similarity: float | None = None,
        user_id: int | None = None,
        notebook_id: int | None = None,
    ) -> AsyncIterator[dict]:
        """Run the streaming RAG pipeline and yield structured WS dict messages.

        Args:
            token: JWT access token presented by the client.
            question: The user's question.
            top_k: Optional override for number of chunks to retrieve.
            min_similarity: Optional override for relevance threshold.
            user_id: Owner filter — retrieval is restricted to this user's
                chunks (tenant isolation; mandatory).
            notebook_id: Optional restriction to a single owned notebook.

        Yields:
            Dicts serialised as JSON WebSocket frames.
        """
        # --- Authentication ---
        try:
            decode_access_token(token)
        except Exception as e:
            msg = WSErrorMessage(detail=f"Authentication failed: {e}").model_dump()
            try:
                yield msg
            except Exception:
                return
            if DEBUG_WS_FLOW:
                logger.debug("WS yielded auth error: %s", msg)
            yield WSCompleteMessage().model_dump()
            return

        # --- Validate question ---
        if not question or not question.strip():
            msg = WSErrorMessage(detail="Question cannot be empty").model_dump()
            try:
                yield msg
            except Exception:
                return
            if DEBUG_WS_FLOW:
                logger.debug("WS yielded empty question error: %s", msg)
            yield WSCompleteMessage().model_dump()
            return

        # --- Retrieval (reused) ---
        try:
            chunks = self.retrieval_service.retrieve(
                question=question,
                top_k=top_k or settings.RAG_TOP_K,
                min_similarity=(
                    settings.RAG_MIN_SIMILARITY
                    if min_similarity is None
                    else min_similarity
                ),
                user_id=user_id,
                notebook_id=notebook_id,
            )
        except ValueError as e:
            msg = WSErrorMessage(detail=f"Invalid question: {e}").model_dump()
            try:
                yield msg
            except Exception:
                return
            if DEBUG_WS_FLOW:
                logger.debug("WS yielded invalid question error: %s", msg)
            yield WSCompleteMessage().model_dump()
            return
        except RuntimeError as e:
            # Raw provider/DB error text stays in the server log; WS clients
            # get a sanitized message.
            logger.error(f"Retrieval failed for question: {e}")
            msg = WSErrorMessage(
                detail="Retrieval is temporarily unavailable. Please try again later."
            ).model_dump()
            try:
                yield msg
            except Exception:
                return
            if DEBUG_WS_FLOW:
                logger.debug("WS yielded retrieval failure error: %s", msg)
            yield WSCompleteMessage().model_dump()
            return

        # --- No relevant context found -> do not call the LLM ---
        if not chunks:
            logger.info(
                "No chunks above min_similarity=%.3f; skipping generation",
                min_similarity or settings.RAG_MIN_SIMILARITY,
            )
            msg = WSContextMessage(
                found_context=False,
                sources=[],
            ).model_dump()
            try:
                yield msg
            except Exception:
                return
            if DEBUG_WS_FLOW:
                logger.debug("WS yielded no-context context message")
            msg = WSCompleteMessage().model_dump()
            try:
                yield msg
            except Exception:
                return
            if DEBUG_WS_FLOW:
                logger.debug("WS yielded completion message")
            return

        # --- Context found: send source info then stream the answer ---
        # Reuse the same source-mapping logic as POST /api/chat so the
        # response shape is consistent (chunk_id, snippet, etc.).
        from app.services.rag import RAGService

        msg = WSContextMessage(
            found_context=True,
            sources=[RAGService.build_source(chunk) for chunk in chunks],
        ).model_dump()
        try:
            yield msg
        except Exception:
            return
        if DEBUG_WS_FLOW:
            logger.debug("WS yielded context message")

        # --- Context found: stream the answer ---
        answer_parts: list[str] = []
        try:
            async for token in self.generation_service.stream_answer(
                question=question, context=self._build_context(chunks)
            ):
                answer_parts.append(token)
                msg = WSAnswerChunkMessage(chunk=token).model_dump()
                try:
                    yield msg
                except Exception:
                    return
                if DEBUG_WS_FLOW:
                    logger.debug("WS yielded answer_chunk (%d chars)", len(token))
        except ValueError as e:
            logger.warning(f"Generation input invalid: {e}")
            msg = WSErrorMessage(detail=f"Generation input invalid: {e}").model_dump()
            try:
                yield msg
            except Exception:
                return
            if DEBUG_WS_FLOW:
                logger.debug("WS yielded generation input error: %s", msg)
            yield WSCompleteMessage().model_dump()
            return
        except Exception as e:
            # GenerationService already sanitizes its RuntimeErrors; any other
            # exception is masked with a generic message here.
            logger.error(f"Generation streaming failed: {e}", exc_info=True)
            detail = (
                str(e)
                if isinstance(e, RuntimeError)
                else "Answer generation is temporarily unavailable. Please try again later."
            )
            msg = WSErrorMessage(detail=detail).model_dump()
            try:
                yield msg
            except Exception:
                return
            if DEBUG_WS_FLOW:
                logger.debug("WS yielded generation failure error: %s", msg)
            yield WSCompleteMessage().model_dump()
            return

        msg = WSCompleteMessage().model_dump()
        try:
            yield msg
        except Exception:
            return
        if DEBUG_WS_FLOW:
            logger.debug("WS yielded completion message")

    def _build_context(self, chunks: list[dict]) -> str:
        """Build a labelled context string from retrieved chunks.

        Kept in sync with ``RAGService.build_context`` so the WebSocket and
        POST ``/api/chat`` endpoints produce identical prompts.
        """
        parts: list[str] = []
        for index, chunk in enumerate(chunks, start=1):
            header = (
                f"[Source {index}] "
                f"(chunk_id={chunk.get('id')}, "
                f"document_id={chunk.get('document_id')}, "
                f"chunk_index={chunk.get('chunk_index')}, "
                f"similarity={chunk.get('similarity_score')})"
            )
            content = (chunk.get("content") or "").strip()
            parts.append(f"{header}\n{content}")

        context = "\n\n".join(parts)

        max_chars = settings.RAG_MAX_CONTEXT_CHARS
        if len(context) > max_chars:
            logger.warning(
                "Context truncated from %d to %d characters",
                len(context),
                max_chars,
            )
            context = context[:max_chars]

        return context
