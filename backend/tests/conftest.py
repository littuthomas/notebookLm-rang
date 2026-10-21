"""
Shared pytest fixtures for endpoint tests.

Strategy:
- Run against a THROWAWAY Postgres schema (not the dev data) by creating a
  unique test schema per session on the configured DATABASE_URL, then tearing
  it down. This keeps tests isolated from real data without needing SQLite
  (pgvector is Postgres-only).
- If no DATABASE_URL is configured, tests that need the DB are skipped.
"""
import os
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-for-pytest-only")

# Security tests must make no external AI calls: use the offline hash-based
# embedding backend and a placeholder Gemini key (satisfies upload plumbing
# checks only). load_dotenv() never overrides values already set in os.environ.
os.environ.setdefault("EMBEDDING_MODEL", "simple-hash")
os.environ.setdefault("GEMINI_API_KEY", "test-key-for-pytest-only-no-real-calls")

# Security tests must make no external AI calls: use the offline hash-based
# embedding backend and a placeholder Gemini key (satisfies upload plumbing
# checks only). load_dotenv() never overrides values already set in os.environ.
os.environ.setdefault("EMBEDDING_MODEL", "simple-hash")
os.environ.setdefault("GEMINI_API_KEY", "test-key-for-pytest-only-no-real-calls")


def _database_url() -> str:
    """Resolve DATABASE_URL from the environment or backend/.env."""
    url = os.environ.get("DATABASE_URL", "")
    if url:
        return url
    # config.py loads backend/.env on import; use it as a fallback
    from app.core.config import get_settings

    return get_settings().DATABASE_URL


@pytest.fixture(scope="session")
def test_engine():
    """Engine bound to a unique per-session schema; dropped afterwards."""
    url = _database_url()
    if not url or url.startswith("postgresql://user:"):
        pytest.skip("DATABASE_URL not configured - skipping DB-backed tests")

    schema = f"test_{uuid.uuid4().hex[:12]}"

    # Neon poolers reject the "options" startup parameter, so tests connect
    # through the DIRECT (unpooled) endpoint. Pin the schema via a connection
    # option rather than a session-level "SET search_path": poolers share
    # server-side connections between clients, so SET leaks across engines.
    direct_url = url.replace("-pooler.", ".")
    sep = "&" if "?" in direct_url else "?"
    schema_url = f"{direct_url}{sep}options=-csearch_path%3D{schema},public"

    # Point the app's global engine at the same test schema BEFORE app modules
    # are imported (env wins over backend/.env because load_dotenv never
    # overrides existing variables). This keeps lifespan side effects
    # (create_all, tenancy migration, seeding) inside the throwaway schema.
    os.environ["DATABASE_URL"] = schema_url

    engine = create_engine(schema_url, pool_pre_ping=True)

    with engine.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA "{schema}"'))

    # Create all tables inside the test schema
    import app.models  # noqa: F401 - register every model

    from app.core.database import Base

    Base.metadata.create_all(bind=engine)

    yield engine

    # Teardown: drop the schema and everything in it
    with engine.begin() as conn:
        conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
    engine.dispose()


@pytest.fixture(scope="session")
def client(test_engine):
    """TestClient wired to the test schema via dependency override."""
    from sqlalchemy.orm import sessionmaker

    TestSession = sessionmaker(bind=test_engine, autocommit=False, autoflush=False)

    from app.core.database import get_db
    from app.main import app

    def override_get_db():
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture(scope="session")
def user_a(client):
    """A registered, logged-in test user. Returns {username, token, headers}."""
    username = f"qa_{uuid.uuid4().hex[:8]}"
    password = "testpass123"
    r = client.post(
        "/api/auth/register",
        json={"username": username, "email": f"{username}@example.com", "password": password},
    )
    assert r.status_code == 201, f"register failed: {r.text}"

    r = client.post("/api/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200, f"login failed: {r.text}"
    token = r.json()["access_token"]

    return {
        "username": username,
        "password": password,
        "token": token,
        "headers": {"Authorization": f"Bearer {token}"},
    }


@pytest.fixture(scope="session")
def user_b(client):
    """A second registered user, used to verify per-user isolation."""
    username = f"qb_{uuid.uuid4().hex[:8]}"
    password = "testpass123"
    r = client.post(
        "/api/auth/register",
        json={"username": username, "email": f"{username}@example.com", "password": password},
    )
    assert r.status_code == 201

    r = client.post("/api/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200
    token = r.json()["access_token"]

    return {
        "username": username,
        "password": password,
        "token": token,
        "headers": {"Authorization": f"Bearer {token}"},
    }
