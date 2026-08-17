# Personal RAG Assistant

API de RAG (Retrieval-Augmented Generation) sobre documentos propios. Sube tus documentos, pregunta en lenguaje natural, recibe respuestas ancladas en su contenido con las fuentes citadas — no un chat genérico que se inventa cosas.

Proyecto de portfolio con foco en **IA aplicada**: no repite auth/CRUD ya demostrados en [`expense-api`](https://github.com/Acedpol/expense-api), se centra en el pipeline real de RAG.

## Pipeline

```
Subir documento → extraer texto (pypdf) → trocear (langchain-text-splitters)
      → generar embeddings (sentence-transformers, local) → indexar en ChromaDB

Pregunta → embeber pregunta → buscar semánticamente en ChromaDB
      → recuperar chunks relevantes → generar respuesta (Claude, o mock si no hay API key)
```

## Stack

- FastAPI + SQLAlchemy (solo metadata de documentos; sin Alembic, deliberado — ver hitos)
- `pypdf` — extracción de texto de PDFs
- `langchain-text-splitters` — chunking real (respeta párrafos/frases, no corta a lo bruto)
- `sentence-transformers` (`all-MiniLM-L6-v2`) — embeddings **locales**, sin coste ni API key
- `chromadb` — vector store embebido
- `anthropic` — generación de respuestas con Claude (opcional, ver abajo)
- pytest, ruff, GitHub Actions

## Sin autenticación (deliberado)

Herramienta de un solo usuario, una única colección de documentos compartida — sin login. El foco de este proyecto es la mecánica de RAG, no repetir JWT por tercera vez (ya demostrado en `expense-api` y usado en `expense-tracker-ui`).

## Arrancar en local

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8010
```

Docs interactivas en `http://localhost:8010/docs`.

## Generación de respuestas: mock vs. Claude real

Sin `ANTHROPIC_API_KEY` configurada, `/ask` usa un **generador simulado honesto**: no inventa una respuesta, muestra literalmente qué fragmentos recuperó y por qué, dejando claro que es una simulación. Todo el resto del pipeline (ingesta, chunking, embeddings, retrieval) es 100% real y gratuito.

En cuanto añadas tu `ANTHROPIC_API_KEY` a `.env`, `/ask` empieza a responder con Claude de verdad, sin tocar código — la selección de proveedor es automática.

## Tests

```bash
pytest -v
```

18 tests contra el pipeline real (embeddings y ChromaDB reales, no mockeados) — mockear la propia mecánica de RAG habría dejado los tests con menos valor que la verificación manual. El único mock es el proveedor de LLM cuando no hay API key, que es exactamente lo que se está probando ahí.

## Estructura

```
app/
├── main.py              # entrypoint, routers, CORS
├── core/                 # config
├── db/                    # base declarativa y sesión (SQLite, solo metadata)
├── models/                 # Document (SQLAlchemy)
├── schemas/                 # Pydantic schemas
├── api/routes/                # documents, search, ask
├── services/                   # document_service (orquesta ingesta)
└── rag/                          # el corazón del proyecto
    ├── text_extraction.py         # pypdf
    ├── chunking.py                  # langchain-text-splitters
    ├── vector_store.py                # embeddings + ChromaDB
    └── llm_provider.py                  # Mock / Anthropic, selección automática
```

## Estado del proyecto / hitos

- [x] Ingesta de documentos (PDF/texto) con extracción real
- [x] Chunking real (no corte fijo ingenuo)
- [x] Embeddings locales + indexado en ChromaDB (métrica coseno)
- [x] Retrieval semántico con score de similitud
- [x] Generación con Claude (real) / mock honesto (sin API key)
- [x] Tests (18, pipeline real) + CI

Pendiente:
- [ ] Deploy real — aplazado, necesita cuenta propia
- [ ] Frontend (repo independiente, mismo patrón que `expense-tracker-ui`)
- [ ] Alembic si el esquema de metadata crece (por ahora `create_all` es suficiente para una tabla)

## Bugs reales encontrados construyendo esto

1. **`pydantic-settings` 2.11** rompe el parseo de `List[str]` desde variables de entorno de una forma que `expense-api` (con 2.5.2) no sufría — mismo patrón de código, versión distinta, comportamiento distinto. Resuelto guardando el valor como `str` y exponiendo la lista vía `@property`.
2. **ChromaDB con métrica de distancia equivocada**: por defecto usa L2 al cuadrado, no coseno — el score de similitud no tenía sentido hasta configurar `hnsw:space: cosine` explícitamente. Detectado calculando el score de verdad, no solo comprobando que `search()` devolvía algo.
3. **Ruta `async def` con código bloqueante**: `sentence-transformers.encode()` corriendo directamente en una ruta `async def` congelaba el *event loop* entero del servidor — ninguna otra petición se atendía mientras tanto, ni siquiera `/health`. Solucionado usando `def` normal (FastAPI paraleliza rutas síncronas en un thread pool automáticamente).
