import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import chat_history_router, chat_router, chat_ws_router, documents_router, notebooks_router, retrieval_router
from app.api.auth import router as auth_router
from app.core.config import get_settings
from app.core.database import engine, Base
from app.models.chunk import DocumentChunk  # noqa: F401 - imported for table registration
from app.utils.seed import seed_sample_data

logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    logger.info("Starting RAG System backend...")

    # Log configuration (without secrets)
    logger.info(f"Embedding model: {settings.EMBEDDING_MODEL}")
    logger.info(f"Embedding dimension: {settings.EMBEDDING_DIMENSION}")
    logger.info(f"Database configured: {'Yes' if settings.is_database_configured else 'No'}")

    # Create database tables if they don't exist
    if settings.is_database_configured and engine is not None:
        logger.info("Creating database tables...")
        try:
            Base.metadata.create_all(bind=engine)
            logger.info("Database tables created successfully")

            # Seed sample data for testing if database is available
            if settings.EMBEDDING_MODEL == "simple-hash":
                logger.info("Seeding sample data for testing...")
                try:
                    seed_sample_data()
                except Exception as e:
                    logger.warning(f"Sample data seeding skipped: {e}")
        except Exception as e:
            logger.warning(f"Database table creation failed: {e}")
            logger.warning("Application will start but retrieval may not work without database")

    yield

    logger.info("Shutting down RAG System backend...")


# Create FastAPI application
app = FastAPI(
    title="RAG System API",
    description="""
Retrieval-Augmented Generation system backend.

## Documents API (ingestion)

1. **POST /api/documents/upload** - Upload a PDF, DOCX, XLSX, TXT, MD or CSV file,
   process it, and store chunks with embeddings
2. **GET /api/documents** - List uploaded documents

## Retrieval API

This API provides the retrieval pipeline for the RAG system:

3. **POST /api/retrieve** - Submit a question and retrieve relevant document chunks
4. **GET /api/retrieve/sample** - Get a sample question for testing

The retrieval pipeline:
1. Takes a user question
2. Converts it to an embedding vector (Gemini Embedding 2, 768 dimensions)
3. Performs vector similarity search using pgvector
4. Returns the most relevant document chunks

## Chat API (end-to-end RAG)

5. **POST /api/chat** - Ask a question and get a Gemini-generated answer grounded in the retrieved chunks

The RAG pipeline:
1. Embeds the question with Gemini Embedding 2
2. Retrieves the Top-K chunks via pgvector cosine similarity
3. Builds a context string from those chunks
4. Asks Gemini to answer using only that context
5. Returns the answer, the sources used, and whether relevant context was found

## Streaming Chat API (WebSocket)

6. **WebSocket /api/chat/ws** - Stream a RAG answer progressively over WebSocket

Same pipeline as POST /api/chat, but the Gemini answer is streamed token-by-token:

1. Client sends a JSON ``start`` message with a JWT ``token`` and a ``question``
2. Server authenticates using the existing JWT decode logic
3. Server retrieves relevant chunks via the existing RetrievalService
4. If no chunk clears the similarity threshold, server sends a ``context`` message
   with ``found_context: false`` then a ``complete`` message (no Gemini call)
5. Otherwise server sends a ``context`` message with the sources and then streams
   ``answer_chunk`` messages until the answer is complete
6. Server sends a final ``complete`` message

Message types (all JSON):
- ``start`` (client -> server)
- ``context`` (server -> client): retrieved sources + found_context flag
- ``answer_chunk`` (server -> client): next answer text fragment
- ``complete`` (server -> client): streaming finished
- ``error`` (server -> client): something went wrong

All WebSocket messages require a valid JWT token (same tokens as POST /api/chat).
    """,
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS - explicit dev origins so the browser preflight always
# succeeds; CORS_ORIGINS env var can override for production.
_cors_origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]
if "*" not in _cors_origins:
    _cors_origins = list(
        dict.fromkeys(_cors_origins + ["http://localhost:5173", "http://127.0.0.1:5173"])
    )
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router)
app.include_router(documents_router)
app.include_router(notebooks_router)
app.include_router(retrieval_router)
app.include_router(chat_router)
app.include_router(chat_history_router)
app.include_router(chat_ws_router)


@app.get("/")
async def root():
    """Root endpoint - health check."""
    return {
        "message": "RAG System API is running",
        "version": "0.1.0",
        "docs_url": "/docs",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
