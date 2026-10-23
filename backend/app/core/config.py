import os

from dotenv import load_dotenv
from pathlib import Path
from functools import lru_cache

# Load .env relative to the backend/ directory (two levels up from this file)
_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(_env_path)


class Settings:
    """Application settings loaded from environment variables."""

    def __init__(self) -> None:
        # Database
        self.DATABASE_URL: str = os.environ.get("DATABASE_URL", "")

        # Embedding
        # Gemini Embedding 2 - the pgvector schema is tied to 768 dimensions.
        self.EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL", "gemini")
        self.EMBEDDING_MODEL_NAME: str = os.environ.get("EMBEDDING_MODEL_NAME", "gemini-embedding-2")
        self.EMBEDDING_DIMENSION: int = int(os.environ.get("EMBEDDING_DIMENSION", "768"))
        self.GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")

        # Gemini LLM (answer generation) - reuses GEMINI_API_KEY above
        self.GEMINI_LLM_MODEL: str = os.environ.get("GEMINI_LLM_MODEL", "gemini-3.8-flash")
        self.GEMINI_LLM_TEMPERATURE: float = float(os.environ.get("GEMINI_LLM_TEMPERATURE", "0.2"))
        self.GEMINI_LLM_MAX_OUTPUT_TOKENS: int = int(os.environ.get("GEMINI_LLM_MAX_OUTPUT_TOKENS", "4096"))

        # Retry policy for TRANSIENT Gemini generation errors (e.g. 503
        # "UNAVAILABLE - high demand", 429 rate limits, 500 internal errors).
        # Retried with exponential backoff + jitter before the OpenRouter
        # fallback is attempted.
        self.GEMINI_LLM_MAX_RETRIES: int = int(os.environ.get("GEMINI_LLM_MAX_RETRIES", "3"))
        self.GEMINI_LLM_RETRY_BASE_DELAY: float = float(
            os.environ.get("GEMINI_LLM_RETRY_BASE_DELAY", "1.0")
        )
        self.GEMINI_LLM_RETRY_MAX_DELAY: float = float(
            os.environ.get("GEMINI_LLM_RETRY_MAX_DELAY", "8.0")
        )

        # Nemotron fallback LLM (generation ONLY; used when Gemini fails).
        # Routed through OpenRouter's OpenAI-compatible API using the
        # OPENROUTER_API_KEY defined below.
        # A smaller, faster model keeps the fallback responsive; the 550B
        # Nemotron Ultra is still available by setting this env var explicitly.
        self.OPENROUTER_LLM_MODEL: str = os.environ.get(
            "OPENROUTER_LLM_MODEL", "meta-llama/llama-3.3-70b-instruct"
        )
        self.OPENROUTER_LLM_BASE_URL: str = os.environ.get(
            "OPENROUTER_LLM_BASE_URL", "https://openrouter.ai/api/v1"
        )
        self.LLM_FALLBACK_ENABLED: bool = os.environ.get(
            "LLM_FALLBACK_ENABLED", "true"
        ).strip().lower() in ("1", "true", "yes", "on")

        # RAG pipeline defaults
        self.RAG_TOP_K: int = int(os.environ.get("RAG_TOP_K", "5"))
        self.RAG_MIN_SIMILARITY: float = float(os.environ.get("RAG_MIN_SIMILARITY", "0.3"))
        self.RAG_MAX_CONTEXT_CHARS: int = int(os.environ.get("RAG_MAX_CONTEXT_CHARS", "12000"))

        # JWT Authentication
        # IMPORTANT: never ship with the default placeholder. Generate a random
        # secret before deployment and set it via the environment (or .env).
        # Backend startup logs a warning when the placeholder is still in use.
        self.JWT_SECRET_KEY: str = os.environ.get("JWT_SECRET_KEY", "changeme-generate-a-real-secret")
        self.JWT_ALGORITHM: str = os.environ.get("JWT_ALGORITHM", "HS256")
        self.ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

        # CORS (frontend/ingestion UI) — must be defined before refresh-cookie
        # defaults that depend on dev-vs-production heuristics.
        self.CORS_ORIGINS: str = os.environ.get("CORS_ORIGINS", "http://localhost:5173")

        # Secure refresh flow (HttpOnly cookie + server-tracked rotation).
        # The refresh cookie lives much longer than the access token and is
        # used only to obtain a new access token when the current one expires.
        self.REFRESH_TOKEN_EXPIRE_DAYS: int = int(os.environ.get("REFRESH_TOKEN_EXPIRE_DAYS", "7"))
        self.REFRESH_COOKIE_NAME: str = os.environ.get("REFRESH_COOKIE_NAME", "rag-refresh")
        self.REFRESH_COOKIE_SECURE: bool = os.environ.get("REFRESH_COOKIE_SECURE", str(self._cookie_bool("REFRESH_COOKIE_SECURE", not self._is_dev))).strip().lower() in ("1", "true", "yes", "on")
        self.REFRESH_COOKIE_SAME_SITE: str = os.environ.get("REFRESH_COOKIE_SAME_SITE", "Lax")

        # Derived cookie config for the browser: when running over HTTPS the
        # only way to send a Secure cookie is to require it, so dev defaults
        # to False and production defaults to True. Local HTTP dev only works
        # with Secure=False (the dev server listens on plain HTTP).
        self.REFRESH_COOKIE_DOMAIN: str = os.environ.get("REFRESH_COOKIE_DOMAIN", "")

        # OpenRouter API key (Nemotron fallback LLM; also an optional embedding
        # backend when EMBEDDING_MODEL=openrouter)
        self.OPENROUTER_API_KEY: str = os.environ.get("OPENROUTER_API_KEY", "")

        # Chunking (document ingestion pipeline)
        self.CHUNK_SIZE: int = int(os.environ.get("CHUNK_SIZE", "1000"))
        self.CHUNK_OVERLAP: int = int(os.environ.get("CHUNK_OVERLAP", "150"))

        # Upload safety limits (DoS / resource protection)
        self.MAX_UPLOAD_SIZE_MB: int = int(os.environ.get("MAX_UPLOAD_SIZE_MB", "20"))
        self.MAX_PAGES_PER_DOC: int = int(os.environ.get("MAX_PAGES_PER_DOC", "500"))
        self.MAX_CHUNKS_PER_DOC: int = int(os.environ.get("MAX_CHUNKS_PER_DOC", "2000"))
        self.MAX_FILENAME_CHARS: int = int(os.environ.get("MAX_FILENAME_CHARS", "255"))

        # Retrieval (ingestion-side alias for RAG_TOP_K)
        self.TOP_K: int = int(os.environ.get("TOP_K", "5"))

        # Application
        self.APP_HOST: str = os.environ.get("APP_HOST", "0.0.0.0")
        self.APP_PORT: int = int(os.environ.get("APP_PORT", "8000"))
        self.DEBUG: bool = os.environ.get("DEBUG", "true").lower() in ("1", "true", "yes")

        if self.JWT_SECRET_KEY == "changeme-generate-a-real-secret":
            import warnings

            warnings.warn(
                "JWT_SECRET_KEY is still the default placeholder value — "
                "generate a real secret before deploying.",
                stacklevel=1,
            )

    # ------------------------------------------------------------------
    # Compatibility aliases (read-only properties, NOT env-backed config)
    #
    # Littu's document-ingestion modules (document_processor, google_embeddings,
    # models/chunk) reference GOOGLE_* settings and google_api_configured.
    # They are mapped onto OUR Gemini Embedding 2 / GEMINI_API_KEY settings so
    # the ingestion pipeline produces 768-dim vectors consistent with the
    # pgvector column and our retrieval pipeline. gemini-embedding-001 and
    # GOOGLE_EMBEDDING_MODEL are intentionally NOT adopted.
    # ------------------------------------------------------------------

    @staticmethod
    def _cookie_bool(env_name: str, dev_default: bool) -> bool:
        """Resolve a boolean cookie-security setting, defaulting to dev behavior."""
        raw = os.environ.get(env_name, "").strip().lower()
        if raw == "":
            return dev_default
        return raw in ("1", "true", "yes", "on")

    @property
    def _is_dev(self) -> bool:
        """Heuristic: treat plain-HTTP localhost/127.0.0.1 as development."""
        origins = [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]
        return any(o.startswith("http://localhost") or o.startswith("http://127.0.0.1") for o in origins)

    @property
    def GOOGLE_API_KEY(self) -> str:
        """Alias for ingestion modules -> GEMINI_API_KEY."""
        return self.GEMINI_API_KEY

    @property
    def GOOGLE_EMBEDDING_MODEL(self) -> str:
        """Alias for ingestion modules -> our EMBEDDING_MODEL_NAME (Gemini Embedding 2)."""
        return self.EMBEDDING_MODEL_NAME

    @property
    def GOOGLE_EMBEDDING_DIMENSION(self) -> int:
        """Alias for ingestion modules -> our EMBEDDING_DIMENSION (768)."""
        return self.EMBEDDING_DIMENSION

    @property
    def GOOGLE_GEMINI_MODEL(self) -> str:
        """Alias for ingestion-side generation -> our GEMINI_LLM_MODEL."""
        return self.GEMINI_LLM_MODEL

    @property
    def google_api_configured(self) -> bool:
        """Check if the Gemini API key is configured (not a placeholder)."""
        return bool(self.GEMINI_API_KEY and not self.GEMINI_API_KEY.startswith("your-"))

    @property
    def use_openrouter(self) -> bool:
        return self.EMBEDDING_MODEL == "openrouter" and self.OPENROUTER_API_KEY

    @property
    def use_gemini(self) -> bool:
        return self.EMBEDDING_MODEL == "gemini"

    @property
    def use_sentence_transformers(self) -> bool:
        return self.EMBEDDING_MODEL == "sentence-transformers"

    @property
    def use_simple_hash(self) -> bool:
        return self.EMBEDDING_MODEL == "simple-hash"

    @property
    def is_database_configured(self) -> bool:
        """Check if database URL is properly configured (not placeholder)."""
        return bool(self.DATABASE_URL and not self.DATABASE_URL.startswith("postgresql://user:"))


@lru_cache
def get_settings() -> Settings:
    return Settings()
