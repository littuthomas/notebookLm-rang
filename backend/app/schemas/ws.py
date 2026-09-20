from pydantic import BaseModel, Field
from typing import Optional


class WSStartMessage(BaseModel):
    """Client -> server: begin a streaming RAG question-answering session."""

    type: str = Field(default="start", description="Message type identifier")
    token: str = Field(
        ...,
        description="JWT access token obtained from POST /api/auth/login",
    )
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The question to answer from the indexed documents",
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=100,
        description="Maximum number of chunks to retrieve (defaults to RAG_TOP_K)",
    )
    min_similarity: Optional[float] = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description=(
            "Minimum cosine similarity for a chunk to count as relevant "
            "(defaults to RAG_MIN_SIMILARITY)"
        ),
    )
    session_id: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Optional chat session id owned by the authenticated user. When "
            "provided, the question and the fully streamed answer are stored "
            "in that session after the complete message. When omitted, the "
            "protocol behaves exactly as before (nothing stored)."
        ),
    )


class WSContextMessage(BaseModel):
    """Server -> client: the retrieved context/source information used for the answer."""

    type: str = Field(default="context", description="Message type identifier")
    found_context: bool = Field(
        ...,
        description=(
            "True if relevant chunks were found and will be used; "
            "False if no chunk cleared the similarity threshold"
        ),
    )
    sources: list[dict] = Field(
        default_factory=list,
        description="Retrieved chunks used as grounding context (same shape as POST /api/chat sources)",
    )


class WSAnswerChunkMessage(BaseModel):
    """Server -> client: a chunk of the progressively generated answer."""

    type: str = Field(default="answer_chunk", description="Message type identifier")
    chunk: str = Field(..., description="Next token/text fragment of the answer")


class WSCompleteMessage(BaseModel):
    """Server -> client: streaming session completed."""

    type: str = Field(default="complete", description="Message type identifier")



class WSErrorMessage(BaseModel):
    """Server -> client: an error occurred during the streaming session."""

    type: str = Field(default="error", description="Message type identifier")
    detail: str = Field(..., description="Human-readable error description")
