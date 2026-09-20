from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request schema for the RAG question-answering endpoint."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The question to answer from the indexed documents",
        examples=["What is the capital of France?"],
    )
    session_id: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Optional chat session id owned by the caller. When provided, the "
            "question and the final answer are stored in that session. When "
            "omitted, the request behaves exactly as before (nothing stored)."
        ),
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="Maximum number of chunks to retrieve (defaults to RAG_TOP_K)",
    )
    min_similarity: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description=(
            "Minimum cosine similarity (0-1) for a chunk to count as relevant "
            "(defaults to RAG_MIN_SIMILARITY)"
        ),
    )


class ChatSource(BaseModel):
    """A retrieved document chunk used as context for the generated answer."""

    chunk_id: int = Field(..., description="Primary key of the document chunk")
    document_id: str = Field(..., description="Identifier of the parent document")
    chunk_index: int = Field(..., description="Position of the chunk within the document")
    similarity_score: float | None = Field(
        default=None, description="Cosine similarity between the question and the chunk"
    )
    snippet: str = Field(..., description="Short excerpt of the chunk content")


class ChatResponse(BaseModel):
    """Response schema for the RAG question-answering endpoint."""

    question: str = Field(..., description="The original question")
    answer: str = Field(..., description="Answer generated from the retrieved context")
    sources: list[ChatSource] = Field(
        default_factory=list, description="Chunks the answer was grounded in"
    )
    found_context: bool = Field(
        ...,
        description=(
            "True if relevant chunks were found and used; False if the answer "
            "could not be found in the available documents"
        ),
    )
    model: str = Field(..., description="Gemini model used for generation")
