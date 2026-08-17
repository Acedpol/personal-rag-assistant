def _upload(client, filename: str, text: str):
    client.post("/documents", files={"file": (filename, text.encode("utf-8"), "text/plain")})


def test_search_finds_the_topically_relevant_document(client):
    _upload(client, "vacaciones.txt", "Los empleados tienen 23 dias de vacaciones al ano.")
    _upload(client, "gastos.txt", "Las dietas de viaje se reembolsan hasta 40 euros al dia.")

    response = client.post(
        "/search", json={"query": "cuantos dias de vacaciones tengo", "top_k": 1}
    )

    assert response.status_code == 200
    results = response.json()
    assert len(results) == 1
    assert "vacaciones" in results[0]["text"].lower()
    assert 0.0 <= results[0]["similarity"] <= 1.0


def test_search_returns_empty_list_when_no_documents(client):
    response = client.post("/search", json={"query": "cualquier cosa"})

    assert response.status_code == 200
    assert response.json() == []


def test_search_respects_top_k(client):
    _upload(client, "a.txt", "Documento sobre vacaciones y permisos laborales.")
    _upload(client, "b.txt", "Documento sobre gastos de viaje y dietas.")
    _upload(client, "c.txt", "Documento sobre teletrabajo y horarios flexibles.")

    response = client.post("/search", json={"query": "politica de la empresa", "top_k": 2})

    assert len(response.json()) == 2
