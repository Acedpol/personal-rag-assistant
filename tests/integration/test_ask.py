from app.core.config import settings


def test_ask_uses_mock_provider_without_api_key(client, monkeypatch):
    # Explicit, not just relying on the test env having no key set — this
    # test's whole point is asserting the mock path, regardless of what a
    # developer's local .env happens to contain.
    monkeypatch.setattr(settings, "anthropic_api_key", None)

    client.post(
        "/documents",
        files={
            "file": (
                "vacaciones.txt",
                "Los empleados tienen 23 dias de vacaciones al ano.".encode("utf-8"),
                "text/plain",
            )
        },
    )

    response = client.post("/ask", json={"question": "cuantos dias de vacaciones tengo"})

    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "MockLLMProvider"
    assert len(body["sources"]) > 0
    assert "Respuesta simulada" in body["answer"]
    # The source sentence must appear complete, not cut mid-phrase.
    assert "Los empleados tienen 23 dias de vacaciones al ano." in body["answer"]


def test_ask_with_no_relevant_documents_returns_empty_sources(client, monkeypatch):
    monkeypatch.setattr(settings, "anthropic_api_key", None)

    response = client.post("/ask", json={"question": "algo"})

    assert response.status_code == 200
    body = response.json()
    assert body["sources"] == []
    assert "No se encontro ningun fragmento" in body["answer"]
