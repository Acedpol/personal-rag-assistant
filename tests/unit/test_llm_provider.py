import pytest

from app.core.config import settings
from app.rag.llm_provider import (
    AnthropicLLMProvider,
    GoogleLLMProvider,
    MockLLMProvider,
    _sentence_safe_excerpt,
    available_providers,
    default_provider,
    get_llm_provider,
)


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


def test_default_provider_prefers_google_when_both_keys_present(monkeypatch):
    monkeypatch.setattr(settings, "google_api_key", "fake-google-key")
    monkeypatch.setattr(settings, "anthropic_api_key", "fake-anthropic-key")

    assert default_provider() == "google"
    assert available_providers() == ["google", "anthropic"]


def test_default_provider_falls_back_to_anthropic_without_google_key(monkeypatch):
    monkeypatch.setattr(settings, "google_api_key", None)
    monkeypatch.setattr(settings, "anthropic_api_key", "fake-anthropic-key")

    assert default_provider() == "anthropic"
    assert available_providers() == ["anthropic"]


def test_default_provider_is_mock_without_any_key(monkeypatch):
    monkeypatch.setattr(settings, "google_api_key", None)
    monkeypatch.setattr(settings, "anthropic_api_key", None)

    assert default_provider() == "mock"
    assert available_providers() == []


def test_get_llm_provider_resolves_the_default_when_no_preference_given(monkeypatch):
    monkeypatch.setattr(settings, "google_api_key", "fake-google-key")
    monkeypatch.setattr(settings, "anthropic_api_key", None)

    assert isinstance(get_llm_provider(), GoogleLLMProvider)


def test_get_llm_provider_honors_an_explicit_available_choice(monkeypatch):
    monkeypatch.setattr(settings, "google_api_key", "fake-google-key")
    monkeypatch.setattr(settings, "anthropic_api_key", "fake-anthropic-key")

    assert isinstance(get_llm_provider("anthropic"), AnthropicLLMProvider)


def test_get_llm_provider_raises_for_an_explicit_unconfigured_choice(monkeypatch):
    monkeypatch.setattr(settings, "google_api_key", "fake-google-key")
    monkeypatch.setattr(settings, "anthropic_api_key", None)

    with pytest.raises(ValueError):
        get_llm_provider("anthropic")
