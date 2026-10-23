from datetime import datetime, timezone, timezone

import sqlalchemy as sa
from sqlalchemy import String, Integer, Text, Boolean, DateTime, ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Notebook(Base):
    """A user's notebook that groups documents, chats, notes and saved answers."""

    __tablename__ = "notebooks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    color: Mapped[str] = mapped_column(String(20), default="lavender", nullable=False)
    is_pinned: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=sa.func.now(), default=_utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=sa.func.now(), default=_utcnow, onupdate=_utcnow, nullable=False
    )

    documents: Mapped[list["DocumentMeta"]] = relationship(
        back_populates="notebook", cascade="all, delete-orphan"
    )
    notes: Mapped[list["Note"]] = relationship(
        back_populates="notebook", cascade="all, delete-orphan"
    )
    saved_answers: Mapped[list["SavedAnswer"]] = relationship(
        back_populates="notebook", cascade="all, delete-orphan"
    )


class DocumentMeta(Base):
    """Metadata for an ingested document (chunks live in document_chunks)."""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    notebook_id: Mapped[int] = mapped_column(Integer, ForeignKey("notebooks.id"), index=True, nullable=False)
    document_uuid: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_type: Mapped[str] = mapped_column(String(20), default="pdf", nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    page_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="ready", nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=sa.func.now(), default=_utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=sa.func.now(), default=_utcnow, onupdate=_utcnow, nullable=False
    )

    notebook: Mapped["Notebook"] = relationship(back_populates="documents")


Index("ix_documents_notebook_created", DocumentMeta.notebook_id, DocumentMeta.created_at)


class Note(Base):
    """A user-written note, optionally linked to a notebook."""

    __tablename__ = "notes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    notebook_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("notebooks.id"), nullable=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    content: Mapped[str] = mapped_column(Text, default="", nullable=False)
    tags: Mapped[str] = mapped_column(String(500), default="", nullable=False)  # comma-separated
    is_pinned: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=sa.func.now(), default=_utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=sa.func.now(), default=_utcnow, onupdate=_utcnow, nullable=False
    )

    notebook: Mapped["Notebook"] = relationship(back_populates="notes")


class SavedAnswer(Base):
    """An AI answer the user chose to save from chat."""

    __tablename__ = "saved_answers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    notebook_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("notebooks.id"), nullable=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    sources_json: Mapped[str] = mapped_column(Text, default="[]", nullable=False)  # JSON array
    source_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=sa.func.now(), default=_utcnow, nullable=False
    )

    notebook: Mapped["Notebook"] = relationship(back_populates="saved_answers")
