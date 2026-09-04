from app.core.config import settings


def test_providers_all_unset_returns_mock_default(client, monkeypatch):
    monkeypatch.setattr(settings, "google_api_key", None)
    monkeypatch.setattr(settings, "anthropic_api_key", None)

    response = client.get("/providers")

    assert response.status_code == 200
    body = response.json()
    assert body["generation"]["default"] == "mock"
    assert body["generation"]["available"] == []
    assert body["embeddings"]["active"] == "local"


def test_providers_google_key_present(client, monkeypatch):
    monkeypatch.setattr(settings, "google_api_key", "fake-google-key")
    monkeypatch.setattr(settings, "anthropic_api_key", None)

    response = client.get("/providers")

    assert response.status_code == 200
    body = response.json()
    assert body["generation"]["default"] == "google"
    assert "google" in body["generation"]["available"]
    assert body["embeddings"]["active"] == "google"


def test_providers_anthropic_only_does_not_activate_google_embeddings(client, monkeypatch):
    monkeypatch.setattr(settings, "google_api_key", None)
    monkeypatch.setattr(settings, "anthropic_api_key", "fake-anthropic-key")

    response = client.get("/providers")

    assert response.status_code == 200
    body = response.json()
    assert body["generation"]["default"] == "anthropic"
    assert body["generation"]["available"] == ["anthropic"]
    # Anthropic has no embeddings API of its own -- its key must never
    # activate Google embeddings.
    assert body["embeddings"]["active"] == "local"
