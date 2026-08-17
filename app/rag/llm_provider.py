from abc import ABC, abstractmethod

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


class MockLLMProvider(LLMProvider):
    """Honest placeholder used when no ANTHROPIC_API_KEY is configured.

    Shows what retrieval actually found instead of pretending to
    synthesize an answer — the retrieval pipeline is real and fully
    testable without any API key; only this last step needs one.
    """

    def generate_answer(self, question: str, context_chunks: list[str]) -> str:
        if not context_chunks:
            return (
                "No se encontro ningun fragmento relevante para responder a "
                f'"{question}". (Generador simulado: configura ANTHROPIC_API_KEY '
                "para respuestas reales de Claude.)"
            )

        preview = "\n\n".join(f"- {chunk[:200]}" for chunk in context_chunks)
        return (
            "[Respuesta simulada -- configura ANTHROPIC_API_KEY para una respuesta "
            f'real de Claude]\n\nSe encontraron {len(context_chunks)} fragmento(s) '
            f'relevante(s) para: "{question}"\n\n{preview}'
        )


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


def get_llm_provider() -> LLMProvider:
    if settings.anthropic_api_key:
        return AnthropicLLMProvider(settings.anthropic_api_key, settings.anthropic_model)
    return MockLLMProvider()
