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
    engine = create_engine(url, pool_pre_ping=True)

    with engine.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA "{schema}"'))

    # Create all tables inside the test schema
    import app.models  # noqa: F401 - register every model

    from app.core.database import Base

    Base.metadata.create_all(bind=engine)

    # Make every future connection use the test schema
    from sqlalchemy import event

    @event.listens_for(engine, "connect")
    def _set_search_path(dbapi_connection, connection_record):  # noqa: ANN001
        cursor = dbapi_connection.cursor()
        cursor.execute(f'SET search_path TO "{schema}"')
        cursor.close()

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
