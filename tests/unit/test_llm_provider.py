from app.rag.llm_provider import MockLLMProvider, _sentence_safe_excerpt


def test_sentence_safe_excerpt_returns_short_text_unchanged():
    assert _sentence_safe_excerpt("Hola mundo.", 200) == "Hola mundo."


def test_sentence_safe_excerpt_cuts_at_sentence_boundary_not_mid_word():
    text = (
        "Los empleados pueden teletrabajar hasta tres dias por semana. "
        "Deben coordinarse con su responsable directo con antelacion suficiente."
    )
    excerpt = _sentence_safe_excerpt(text, 70)

    assert excerpt == "Los empleados pueden teletrabajar hasta tres dias por semana."
    assert excerpt[-1] in ".!?"


def test_sentence_safe_excerpt_falls_back_to_word_boundary_without_sentence_end():
    text = "palabra " * 40  # no punctuation anywhere
    excerpt = _sentence_safe_excerpt(text, 50)

    assert not excerpt.endswith("palabr")
    assert excerpt.endswith("…")


def test_mock_provider_leads_with_full_top_chunk_not_a_bullet_dump():
    provider = MockLLMProvider()
    chunk = "Los empleados tienen veintitres dias de vacaciones al ano, acumulados de forma proporcional."

    answer = provider.generate_answer("cuantos dias de vacaciones tengo", [chunk])

    assert chunk in answer
    assert "Otros fragmentos relevantes" not in answer


def test_mock_provider_lists_additional_distinct_chunks_sentence_safe():
    provider = MockLLMProvider()
    top = "Los empleados tienen veintitres dias de vacaciones al ano."
    second = "palabra " * 40  # long enough to force a cut, no punctuation

    answer = provider.generate_answer("cuantos dias de vacaciones tengo", [top, second])

    assert "Otros fragmentos relevantes" in answer
    assert not answer.rstrip().endswith("palabr")
