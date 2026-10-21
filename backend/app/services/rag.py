import logging
from typing import Optional

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.services.generation import (
    NO_ANSWER_MESSAGE,
    GenerationService,
    get_generation_service,
)
from app.services.retrieval import RetrievalService, create_retrieval_service

logger = logging.getLogger(__name__)
settings = get_settings()

# How much of each chunk's content to expose in the response "sources" list.
SNIPPET_MAX_CHARS = 300


class RAGService:
    """End-to-end RAG pipeline.

    Composes the existing retrieval service (Gemini Embedding 2 + pgvector cosine
    similarity) with the Gemini generation service:

        question -> embedding -> top-K chunks -> context -> LLM answer
    """

    def __init__(
        self,
        retrieval_service: RetrievalService,
        generation_service: Optional[GenerationService] = None,
        max_context_chars: Optional[int] = None,
    ) -> None:
        self.retrieval_service = retrieval_service
        self.generation_service = generation_service or get_generation_service()
        self.max_context_chars = max_context_chars or settings.RAG_MAX_CONTEXT_CHARS

    def answer_question(
        self,
        question: str,
        top_k: Optional[int] = None,
        min_similarity: Optional[float] = None,
        user_id: Optional[int] = None,
        notebook_id: Optional[int] = None,
    ) -> dict:
        """
        Answer a question using retrieved document chunks as context.

        Args:
            question: The user's question.
            top_k: Maximum number of chunks to retrieve (defaults to RAG_TOP_K).
            min_similarity: Minimum cosine similarity (defaults to RAG_MIN_SIMILARITY).
            user_id: Owner filter — retrieval is restricted to this user's
                chunks (tenant isolation; mandatory).
            notebook_id: Optional restriction to a single owned notebook.

        Returns:
            Dict with ``answer``, ``sources``, ``found_context`` and ``model``.

        Raises:
            ValueError: If the question is empty.
            RuntimeError: If retrieval or generation fails.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        resolved_top_k = top_k or settings.RAG_TOP_K
        resolved_min_similarity = (
            settings.RAG_MIN_SIMILARITY if min_similarity is None else min_similarity
        )

        # Steps 1-3: query embedding + pgvector cosine-similarity search (reused)
        # Scoped to the authenticated user AND, when given, a single notebook.
        chunks = self.retrieval_service.retrieve(
            question=question,
            top_k=resolved_top_k,
            min_similarity=resolved_min_similarity,
            user_id=user_id,
            notebook_id=notebook_id,
        )

        # Step 4: no sufficiently relevant chunks -> do not call the LLM / hallucinate
        if not chunks:
            logger.info(
                "No chunks above min_similarity=%.3f (user_id=%s); skipping generation",
                resolved_min_similarity,
                user_id,
            )
            return {
                "answer": NO_ANSWER_MESSAGE,
                "sources": [],
                "found_context": False,
                # No LLM was called, so report the configured primary model.
                "model": self.generation_service.model_name,
            }

        # Step 5: build the grounded context string
        context = self.build_context(chunks)

        # Steps 6-7: ask Gemini to answer using only the supplied context
        # (falls back to Nemotron Ultra via OpenRouter if the Gemini call fails).
        answer = self.generation_service.generate_answer(
            question=question,
            context=context,
        )

        # Report the model that actually produced the answer (Gemini, or
        # Nemotron when the fallback fired).
        model_name = self.generation_service.last_model_used

        return {
            "answer": answer,
            "sources": [self.build_source(c) for c in chunks],
            "found_context": True,
            "model": model_name,
        }

    def build_context(self, chunks: list[dict]) -> str:
        """Turn retrieved chunks into a labelled context string for the LLM."""
        parts = []
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

        # Guard against oversized prompts
        if len(context) > self.max_context_chars:
            logger.warning(
                "Context truncated from %d to %d characters",
                len(context),
                self.max_context_chars,
            )
            context = context[: self.max_context_chars]

        return context

    @staticmethod
    def build_source(chunk: dict) -> dict:
        """Map a retrieved chunk to the response ``sources`` entry."""
        content = (chunk.get("content") or "").strip()
        snippet = content[:SNIPPET_MAX_CHARS]
        if len(content) > SNIPPET_MAX_CHARS:
            snippet += "..."

        return {
            "chunk_id": chunk.get("id"),
            "document_id": chunk.get("document_id"),
            "chunk_index": chunk.get("chunk_index"),
            "similarity_score": chunk.get("similarity_score"),
            "snippet": snippet,
        }


def create_rag_service(
    db: Session,
    generation_service: Optional[GenerationService] = None,
) -> RAGService:
    """Factory function to create a RAG service with a database session."""
    return RAGService(
        retrieval_service=create_retrieval_service(db),
        generation_service=generation_service,
    )
