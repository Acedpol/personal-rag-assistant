from pathlib import Path

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def test_upload_text_document(client):
    response = client.post(
        "/documents", files={"file": ("test.txt", b"Contenido de prueba.", "text/plain")}
    )

    assert response.status_code == 201
    body = response.json()
    assert body["filename"] == "test.txt"
    assert body["chunk_count"] == 1


def test_upload_real_pdf_extracts_text(client):
    pdf_bytes = (FIXTURES_DIR / "sample.pdf").read_bytes()

    response = client.post(
        "/documents", files={"file": ("sample.pdf", pdf_bytes, "application/pdf")}
    )

    assert response.status_code == 201
    assert response.json()["char_count"] > 0


def test_upload_rejects_unsupported_content_type(client):
    response = client.post(
        "/documents", files={"file": ("test.json", b"{}", "application/json")}
    )
    assert response.status_code == 400


def test_list_and_get_document(client):
    create_response = client.post(
        "/documents", files={"file": ("a.txt", b"Hola", "text/plain")}
    )
    document_id = create_response.json()["id"]

    list_response = client.get("/documents")
    assert len(list_response.json()) == 1

    detail_response = client.get(f"/documents/{document_id}")
    assert detail_response.json()["extracted_text"] == "Hola"


def test_get_document_404_when_missing(client):
    response = client.get("/documents/999")
    assert response.status_code == 404


def test_delete_document(client):
    create_response = client.post(
        "/documents", files={"file": ("a.txt", b"Hola", "text/plain")}
    )
    document_id = create_response.json()["id"]

    delete_response = client.delete(f"/documents/{document_id}")
    assert delete_response.status_code == 204

    get_response = client.get(f"/documents/{document_id}")
    assert get_response.status_code == 404


def test_preview_document_chunks(client):
    create_response = client.post(
        "/documents", files={"file": ("a.txt", b"Primer parrafo.\n\nSegundo parrafo.", "text/plain")}
    )
    document_id = create_response.json()["id"]

    response = client.get(f"/documents/{document_id}/chunks")

    assert response.status_code == 200
    chunks = response.json()
    assert len(chunks) == 1
    assert "Primer parrafo" in chunks[0]["text"]
