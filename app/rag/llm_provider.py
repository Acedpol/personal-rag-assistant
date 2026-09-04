from abc import ABC, abstractmethod
from typing import Optional

from app.core.config import settings

RAG_SYSTEM_PROMPT = (
    "Responde a la pregunta del usuario basandote UNICAMENTE en el contexto "
    "proporcionado. Si el contexto no contiene la respuesta, dilo explicitamente "
    "en vez de inventar informacion."
)


class LLMProvider(ABC):
    @abstractmethod
    def generate_answer(self, question: str, context_chunks: list[str]) -> str:
        raise NotImplementedError


def _sentence_safe_excerpt(text: str, max_chars: int) -> str:
    """Shortens text to at most max_chars without cutting mid-sentence/mid-word.

    Looks for the last sentence boundary (". ", "! ", "? ", or a paragraph
    break) at or before max_chars and cuts there. Falls back to the last
    whitespace (never mid-word) if no sentence boundary is found.
    """
    text = text.strip()
    if len(text) <= max_chars:
        return text

    window = text[:max_chars]
    for sep in (". ", "! ", "? ", "\n\n"):
        idx = window.rfind(sep)
        if idx != -1:
            return text[: idx + 1].strip()

    idx = window.rfind(" ")
    if idx != -1:
        return text[:idx].strip() + "…"
    return window + "…"


class MockLLMProvider(LLMProvider):
    """Honest placeholder used when no LLM API key is configured.

    Shows what retrieval actually found instead of pretending to
    synthesize an answer — the retrieval pipeline is real and fully
    testable without any API key; only this last step needs one.
    """

    def generate_answer(self, question: str, context_chunks: list[str]) -> str:
        if not context_chunks:
            return (
                "No se encontro ningun fragmento relevante para responder a "
                f'"{question}". (Generador simulado: configura GOOGLE_API_KEY o '
                "ANTHROPIC_API_KEY para una respuesta real.)"
            )

        # Lead with the most relevant chunk (already ranked by vector_search)
        # in full, sentence-safe if it's long -- one coherent excerpt reads
        # like an answer, unlike a bullet dump of several truncated ones.
        lead = _sentence_safe_excerpt(context_chunks[0], 600)
        answer = (
            "[Respuesta simulada -- configura GOOGLE_API_KEY o ANTHROPIC_API_KEY "
            f"para una respuesta real]\n\n{lead}"
        )

        others = [
            _sentence_safe_excerpt(chunk, 200)
            for chunk in context_chunks[1:]
            if chunk.strip() and chunk.strip() != context_chunks[0].strip()
        ]
        if others:
            bullets = "\n\n".join(f"- {excerpt}" for excerpt in others)
            answer += f"\n\nOtros fragmentos relevantes:\n\n{bullets}"

        return answer


class AnthropicLLMProvider(LLMProvider):
    def __init__(self, api_key: str, model: str):
        from anthropic import Anthropic

        self._client = Anthropic(api_key=api_key)
        self._model = model

    def generate_answer(self, question: str, context_chunks: list[str]) -> str:
        if not context_chunks:
            return "No se encontro ningun fragmento relevante para responder a esa pregunta."

        context = "\n\n---\n\n".join(context_chunks)
        prompt = f"{RAG_SYSTEM_PROMPT}\n\nContexto:\n{context}\n\nPregunta: {question}"

        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text


class GoogleLLMProvider(LLMProvider):
    def __init__(self, api_key: str, model: str):
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._model = model

    def generate_answer(self, question: str, context_chunks: list[str]) -> str:
        if not context_chunks:
            return "No se encontro ningun fragmento relevante para responder a esa pregunta."

        context = "\n\n---\n\n".join(context_chunks)
        prompt = f"{RAG_SYSTEM_PROMPT}\n\nContexto:\n{context}\n\nPregunta: {question}"

        response = self._client.models.generate_content(model=self._model, contents=prompt)
        return response.text


def available_providers() -> list[str]:
    available = []
    if settings.google_api_key:
        available.append("google")
    if settings.anthropic_api_key:
        available.append("anthropic")
    return available


def default_provider() -> str:
    available = available_providers()
    if "google" in available:
        return "google"
    if "anthropic" in available:
        return "anthropic"
    return "mock"


def get_llm_provider(preferred: Optional[str] = None) -> LLMProvider:
    choice = preferred or default_provider()

    if choice == "google":
        if not settings.google_api_key:
            raise ValueError("google provider requested but GOOGLE_API_KEY is not configured")
        return GoogleLLMProvider(settings.google_api_key, settings.google_generation_model)

    if choice == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError("anthropic provider requested but ANTHROPIC_API_KEY is not configured")
        return AnthropicLLMProvider(settings.anthropic_api_key, settings.anthropic_model)

    return MockLLMProvider()
