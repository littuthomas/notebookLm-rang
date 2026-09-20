"""Pydantic schemas for chat session/history endpoints."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class MessageOut(BaseModel):
    """One stored message (question or answer) in a chat session."""

    id: int = Field(..., description="Primary key of the stored message")
    role: str = Field(..., description='Message author role: "user" or "assistant"')
    content: str = Field(..., description="Full message text")
    created_at: datetime = Field(..., description="When the message was stored")

    model_config = ConfigDict(from_attributes=True)


class SessionSummary(BaseModel):
    """Summary of a chat session (without messages), for create/list views."""

    id: int = Field(..., description="Primary key of the chat session")
    title: str = Field(..., description="Session title (derived from the first question)")
    created_at: datetime = Field(..., description="When the session was created")
    updated_at: datetime = Field(..., description="When the last exchange was recorded")
    message_count: int = Field(
        ..., ge=0, description="Number of stored messages in this session"
    )

    model_config = ConfigDict(from_attributes=True)


class SessionDetail(BaseModel):
    """A single chat session including its messages, for the detail view."""

    id: int = Field(..., description="Primary key of the chat session")
    title: str = Field(..., description="Session title (derived from the first question)")
    created_at: datetime = Field(..., description="When the session was created")
    updated_at: datetime = Field(..., description="When the last exchange was recorded")
    messages: list[MessageOut] = Field(
        default_factory=list,
        description="Stored messages ordered by creation time",
    )

    model_config = ConfigDict(from_attributes=True)


class SessionListResponse(BaseModel):
    """Response for GET /api/chat/sessions."""

    sessions: list[SessionSummary] = Field(
        default_factory=list, description="The caller's sessions, newest activity first"
    )
    total: int = Field(..., ge=0, description="Total number of sessions returned")


class DeleteSessionResponse(BaseModel):
    """Response for DELETE /api/chat/sessions/{session_id}."""

    deleted: bool = Field(True, description="Confirmation that the session was removed")
    session_id: int = Field(..., description="Id of the removed session")
    messages_deleted: int = Field(
        ..., ge=0, description="Number of messages removed with the session"
    )
