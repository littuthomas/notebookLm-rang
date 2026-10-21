from sqlalchemy import String, Integer, Text, ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column
from pgvector.sqlalchemy import Vector
from app.core.database import Base
from app.core.config import get_settings

settings = get_settings()


class DocumentChunk(Base):
    """Represents a processed document chunk with its embedding vector.

    Every chunk is owned by the user who uploaded the parent document and is
    linked to that user's notebook. Both columns are nullable so pre-existing
    rows (created before this column existed) keep loading; the runtime
    migration in app/main.py backfills them from the documents table, and
    retrieval IGNORES rows whose user_id is still NULL so legacy/unowned data
    can never leak between users.
    """

    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    document_id: Mapped[str] = mapped_column(String, index=True)
    chunk_index: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[Vector] = mapped_column(Vector(settings.GOOGLE_EMBEDDING_DIMENSION))  # Google text-embedding-004 = 768
    # --- Tenant isolation columns (owner of the parent document) ---
    user_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("users.id"), index=True, nullable=True
    )
    notebook_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("notebooks.id"), index=True, nullable=True
    )

    def to_dict(self, similarity_score: float | None = None) -> dict:
        """Convert chunk to dictionary for API response."""
        result: dict = {
            "id": self.id,
            "document_id": self.document_id,
            "chunk_index": self.chunk_index,
            "content": self.content,
        }
        if similarity_score is not None:
            result["similarity_score"] = similarity_score
        return result


# Cover the hot path: "vector search restricted to one owner".
Index("ix_document_chunks_user_id", DocumentChunk.user_id)
