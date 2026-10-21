import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.chunk import DocumentChunk
from app.services.embedding import get_embedding_service

logger = logging.getLogger(__name__)


class RetrievalService:
    """Service for retrieving relevant document chunks based on query."""

    def __init__(self, db: Session):
        self.db = db
        self.embedding_service = get_embedding_service()

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        min_similarity: float = 0.0,
        user_id: int | None = None,
        notebook_id: int | None = None,
    ) -> list[dict]:
        """
        Retrieve the most relevant document chunks for a given question.

        Args:
            question: The user's question to search for.
            top_k: Maximum number of chunks to return.
            min_similarity: Minimum similarity score (0-1) for results.
            user_id: Owner filter (tenant isolation). Chunks from other users
                and unowned legacy chunks are NEVER considered.
            notebook_id: Optional additional filter to a single notebook.

        Returns:
            List of retrieved chunks with similarity scores.

        Raises:
            ValueError: If the question is empty or embedding dimension mismatches.
            RuntimeError: If embedding or search fails.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
        # Tenant isolation is mandatory: without an owner there is no safe
        # scope, so refuse instead of falling back to a global search.
        if user_id is None:
            raise ValueError("user_id is required for retrieval (data isolation)")

        try:
            # Step 1: Generate query embedding
            query_embedding = self.embedding_service.embed(question)
            logger.debug(
                f"Generated query embedding with {len(query_embedding)} dimensions"
            )

            # Step 2: Validate embedding dimension matches the model's expected dimension
            expected_dim = self.embedding_service.embedding_dim
            if len(query_embedding) != expected_dim:
                raise ValueError(
                    f"Query embedding dimension {len(query_embedding)} "
                    f"does not match expected dimension {expected_dim}"
                )

            # Step 3: Perform vector similarity search using pgvector,
            # restricted to chunks owned by the requesting user.
            results = self._vector_search(
                query_embedding=query_embedding,
                top_k=top_k,
                min_similarity=min_similarity,
                user_id=user_id,
                notebook_id=notebook_id,
            )

            logger.info(f"Retrieved {len(results)} chunks for question")
            return results

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise RuntimeError(f"Failed to retrieve chunks: {e}") from e

    def _vector_search(
        self,
        query_embedding: list[float],
        top_k: int,
        min_similarity: float,
        user_id: int | None,
        notebook_id: int | None = None,
    ) -> list[dict]:
        """
        Perform vector similarity search using pgvector.

        Uses cosine distance (<=> operator) for similarity, which is the
        standard metric for text embeddings produced by sentence-transformers.

        pgvector's cosine_distance returns a value between 0 (identical) and
        2 (opposite). We convert to cosine_similarity = 1 - cosine_distance,
        giving a range of -1 to 1 where 1 means identical.

        Args:
            query_embedding: The query vector as a list of floats.
            top_k: Maximum number of results to return.
            min_similarity: Minimum cosine similarity score to include.
            user_id: Owner filter (mandatory, see retrieve()).
            notebook_id: Optional notebook restriction.

        Returns:
            List of dicts with chunk data and similarity scores.
        """
        # pgvector works directly with Python lists — no numpy needed
        # Select distance alongside chunks to avoid re-computing per row
        distance_col = DocumentChunk.embedding.cosine_distance(query_embedding).label(
            "cosine_distance"
        )

        stmt = select(DocumentChunk, distance_col)

        # --- Tenant isolation (applied in SQL, never in Python) ---
        # Only chunks owned by the requesting user can ever match. Rows with
        # a NULL user_id (legacy/unowned data) are excluded as well.
        stmt = stmt.where(DocumentChunk.user_id == user_id)
        if notebook_id is not None:
            stmt = stmt.where(DocumentChunk.notebook_id == notebook_id)

        stmt = stmt.order_by(distance_col).limit(top_k)

        try:
            rows = self.db.execute(stmt).all()
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise RuntimeError(f"Vector search failed: {e}") from e

        results = []
        for chunk, cosine_distance in rows:
            # Convert cosine distance to cosine similarity
            # cosine_distance = 1 - cosine_similarity
            # so cosine_similarity = 1 - cosine_distance
            cosine_similarity = 1.0 - float(cosine_distance)

            # Apply minimum similarity filter
            if cosine_similarity >= min_similarity:
                results.append(
                    chunk.to_dict(similarity_score=round(cosine_similarity, 4))
                )

        return results


def create_retrieval_service(db: Session) -> RetrievalService:
    """Factory function to create a retrieval service with a database session."""
    return RetrievalService(db=db)
