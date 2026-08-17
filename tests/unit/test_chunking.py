from app.core.config import settings
from app.rag.chunking import split_text


def test_split_text_returns_single_chunk_for_short_text():
    assert split_text("Texto corto.") == ["Texto corto."]


def test_split_text_splits_long_text_into_multiple_chunks():
    long_text = "Parrafo de relleno para forzar el corte. " * 100
    chunks = split_text(long_text)

    assert len(chunks) > 1
    assert all(len(chunk) <= settings.chunk_size for chunk in chunks)


def test_split_text_prefers_paragraph_boundaries():
    text = "Primer parrafo con contenido relevante.\n\nSegundo parrafo, tambien relevante."
    chunks = split_text(text)
    # con un chunk_size grande, ambos parrafos caben en un unico chunk
    assert len(chunks) == 1
    assert "Primer parrafo" in chunks[0]
    assert "Segundo parrafo" in chunks[0]
