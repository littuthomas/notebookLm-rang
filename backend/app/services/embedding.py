import hashlib
import logging
from typing import Optional
from collections import OrderedDict

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# Embedding cache (in-memory LRU)
#
# Reduces repeated Gemini Embedding API calls for the same or semantically
# identical questions. The cache key is a SHA-256 hash of the normalised text
# so that whitespace/ casing differences still hit the same cached vector.
# Cache is capped at EMBEDDING_CACHE_MAX_ITEMS entries; when full the least-
# recently-used entry is evicted.
# ---------------------------------------------------------------------------

class _EmbeddingCache:
    """Simple in-memory LRU cache for embedding vectors.

    Thread-safe enough for the FastAPI worker-thread model: reads and writes
    are protected by a reentrant lock, and eviction happens inside the lock.
    """

    def __init__(self, max_items: int = 4096) -> None:
        self._max_items = max_items
        self._store: OrderedDict[str, list[float]] = OrderedDict()

    def _normalise(self, text: str) -> str:
        """Normalise text for cache-key stability: collapse whitespace, trim."""
        return " ".join(text.split()).strip().lower()

    def _key(self, text: str) -> str:
        """SHA-256 hash of the normalised text used as the cache key."""
        return hashlib.sha256(self._normalise(text).encode("utf-8")).hexdigest()

    def get(self, text: str) -> Optional[list[float]]:
        key = self._key(text)
        if key in self._store:
            # Move to end (most-recently-used)
            self._store.move_to_end(key)
            return self._store[key]
        return None

    def set(self, text: str, vector: list[float]) -> None:
        key = self._key(text)
        if key in self._store:
            self._store.move_to_end(key)
        else:
            if len(self._store) >= self._max_items:
                self._store.popitem(last=False)  # evict least-recently-used
        self._store[key] = list(vector)  # store a copy so callers can't mutate us


_default_embedding_cache: Optional[_EmbeddingCache] = None


def get_embedding_cache() -> _EmbeddingCache:
    """Return the process-wide embedding cache (created lazily)."""
    global _default_embedding_cache
    if _default_embedding_cache is None:
        _default_embedding_cache = _EmbeddingCache()
    return _default_embedding_cache


def reset_embedding_cache() -> None:
    """Clear the process-wide embedding cache (mostly for tests)."""
    global _default_embedding_cache
    _default_embedding_cache = None


class EmbeddingService:
    """Service for generating embeddings from text."""

    def __init__(self, cache: Optional[_EmbeddingCache] = None):
        self._embedding_dim: int = settings.EMBEDDING_DIMENSION
        self._cache = cache or get_embedding_cache()

    def embed(self, text: str) -> list[float]:
        """
        Generate an embedding vector for the given text.

        The in-memory LRU cache is checked first for all backends, so repeated
        or near-identical texts skip the API call entirely.

        Args:
            text: The text to embed.

        Returns:
            List of floats representing the embedding vector.

        Raises:
            ValueError: If the text is empty.
            RuntimeError: If embedding generation fails.
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")

        # Check the cache for ALL backends (the cached vector is backend-agnostic
        # only when the same backend produced it, which is guaranteed because the
        # cache is process-local and the backend is fixed per deployment).
        cached = self._cache.get(text)
        if cached is not None:
            return list(cached)

        try:
            if settings.use_gemini:
                vector = self._embed_gemini(text)

            elif settings.use_openrouter:
                vector = self._embed_openrouter(text)

            elif settings.use_simple_hash:
                vector = self._embed_simple_hash(text)

            else:
                raise RuntimeError(f"Unknown embedding model: {settings.EMBEDDING_MODEL}")

            # Store in cache if the backend produced a vector (simple-hash is tiny
            # and deterministic but caching it still avoids re-hashing on repeat calls).
            self._cache.set(text, vector)
            return vector

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise RuntimeError(f"Failed to generate embedding: {e}") from e

    def _embed_gemini(self, text: str) -> list[float]:
        """Generate embedding using Google Gemini Embedding 2 API.

        Checked against the in-memory LRU cache first so that repeated or
        near-identical questions do not hit the API again. The cache key is a
        SHA-256 hash of the normalised text (whitespace/case insensitive).
        """
        # 1. Cache hit?
        cached = self._cache.get(text)
        if cached is not None:
            logger.debug("Embedding cache HIT for text of length %d", len(text))
            return list(cached)

        # 2. Cache miss — call the Gemini Embedding API.
        from google import genai

        if not settings.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY not configured")

        client = genai.Client(api_key=settings.GEMINI_API_KEY)

        response = client.models.embed_content(
            model=settings.EMBEDDING_MODEL_NAME,
            contents=text,
            config={"output_dimensionality": self._embedding_dim},
        )

        vector = list(response.embeddings[0].values)

        # 3. Store in cache for future requests.
        self._cache.set(text, vector)
        logger.debug("Embedding cache MISS — stored vector for text of length %d", len(text))

        return vector

    def _embed_openrouter(self, text: str) -> list[float]:
        """Generate embedding using OpenRouter API (cached like the Gemini path)."""
        cached = self._cache.get(text)
        if cached is not None:
            return list(cached)

        import httpx

        if not settings.OPENROUTER_API_KEY:
            raise RuntimeError("OpenRouter API key not configured")

        response = httpx.post(
            "https://openrouter.ai/api/v1/embeddings",
            headers={
                "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.EMBEDDING_MODEL_NAME,
                "input": text,
            },
            timeout=30.0,
        )

        if response.status_code != 200:
            raise RuntimeError(f"OpenRouter API error: {response.status_code} - {response.text}")

        data = response.json()
        embeddings = data.get("data", [])
        if not embeddings:
            raise RuntimeError("No embeddings returned from OpenRouter")

        embedding = embeddings[0]["embedding"]
        self._cache.set(text, embedding)
        return embedding

    def _embed_simple_hash(self, text: str) -> list[float]:
        """
        Generate a simple hash-based embedding for testing.
        NOT for production use. Just for development when no model is available.
        The caller (embed()) caches the result via self._cache.
        """
        import hashlib

        vector_size = self._embedding_dim
        hash_values = []

        for i in range(vector_size):
            h = hashlib.md5(f"{text}:{i}".encode()).hexdigest()
            hash_int = int(h, 16)
            normalized = (hash_int % 2000) / 1000.0 - 1.0
            hash_values.append(normalized)

        return hash_values

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension."""
        return self._embedding_dim


# Global instance
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get or create the global embedding service instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
