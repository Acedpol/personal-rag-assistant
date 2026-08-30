import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.db.base import Base
from app.db.session import get_db
from app.main import app
from app.rag import vector_store

TEST_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture()
def db_session():
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def isolated_vector_store(tmp_path, monkeypatch):
    # Point Chroma at a fresh temp dir per test and drop the cached client
    # so it actually reconnects there — the embedding model's own cache is
    # deliberately left alone (it's stateless and expensive to reload; no
    # test-isolation reason to clear it).
    monkeypatch.setattr(settings, "chroma_persist_dir", str(tmp_path / "chroma"))
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path / "documents"))
    # Default every client-based test to the local embedding provider,
    # regardless of what a developer's real .env contains -- without this,
    # the first person to configure a real GOOGLE_API_KEY locally has every
    # /documents upload and /ask call in the suite silently start hitting
    # the real Google API (slow, costs quota, and non-deterministic).
    # Found exactly that way: test_ask.py's mock-path tests only neutralized
    # anthropic_api_key, and started making real Gemini calls the moment a
    # real Google key existed. Tests that want the Google path explicitly
    # override this back in their own body (see test_providers.py).
    monkeypatch.setattr(settings, "google_api_key", None)
    vector_store.get_chroma_client.cache_clear()
    yield
    vector_store.get_chroma_client.cache_clear()


@pytest.fixture()
def client(db_session, isolated_vector_store):
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
