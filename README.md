# Personal RAG Assistant

API de RAG (Retrieval-Augmented Generation) sobre documentos propios. Sube tus documentos, pregunta en lenguaje natural, recibe respuestas ancladas en su contenido con las fuentes citadas — no un chat genérico que se inventa cosas.

Proyecto de portfolio con foco en **IA aplicada**: no repite auth/CRUD ya demostrados en [`expense-api`](https://github.com/Acedpol/expense-api), se centra en el pipeline real de RAG.

## Pipeline

```
Subir documento → extraer texto (pypdf) → trocear (langchain-text-splitters)
      → generar embeddings (Google si hay API key, si no local) → indexar en ChromaDB

Pregunta → embeber pregunta (mismo proveedor que la indexación) → buscar semánticamente
      → recuperar chunks relevantes → generar respuesta (Google Gemini por defecto,
        Claude si se selecciona explícitamente, o mock si no hay ninguna API key)
```

## Stack

- FastAPI + SQLAlchemy (solo metadata de documentos; sin Alembic, deliberado — ver hitos)
- `pypdf` — extracción de texto de PDFs
- `langchain-text-splitters` — chunking real (respeta párrafos/frases, no corta a lo bruto)
- `sentence-transformers` (`all-MiniLM-L6-v2`) — embeddings **locales**, sin coste ni API key, fallback sin configuración
- `google-genai` — generación con Gemini (proveedor por defecto) y embeddings de Google, ambos tras la misma `GOOGLE_API_KEY`
- `chromadb` — vector store embebido, una colección por proveedor+dimensión de embedding
- `anthropic` — generación de respuestas con Claude, alternativa explícita a Gemini (ver abajo)
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

## Generación de respuestas: Google Gemini, Claude o simulado

Tres niveles, en este orden de preferencia:

1. **Google Gemini** (`GOOGLE_API_KEY` configurada) — proveedor **por defecto**, gratuito en el nivel de AI Studio.
2. **Claude** (`ANTHROPIC_API_KEY` configurada) — se usa si se selecciona explícitamente, o si no hay key de Google.
3. **Generador simulado honesto** — si no hay ninguna key: no inventa una respuesta, muestra el fragmento más relevante completo (nunca cortado a mitad de frase) y deja claro que es una simulación.

Todo el resto del pipeline (ingesta, chunking, embeddings, retrieval) es 100% real y gratuito sin ninguna key.

`POST /ask` acepta un campo opcional `provider: "google" | "anthropic"` para forzar uno de los dos explícitamente en esa pregunta concreta; si se pide uno sin su key configurada, devuelve `400` en vez de caer en silencio a mock. `GET /providers` expone el proveedor por defecto, los disponibles, y el proveedor de embeddings activo — así un cliente (como el frontend) sabe de antemano qué opciones ofrecer.

## Embeddings: locales o Google

Igual criterio que la generación: si hay `GOOGLE_API_KEY`, la ingesta y la búsqueda usan el modelo de embeddings de Google; si no, `sentence-transformers` local. La key de Anthropic nunca activa embeddings de Google — Anthropic no tiene una API de embeddings propia.

ChromaDB no permite mezclar embeddings de distinta dimensión en una misma colección, así que cada proveedor+dimensión tiene su propia colección (`document_chunks_local_384d`, `document_chunks_google_768d`). Cambiar de proveedor de embeddings (p. ej. añadir la key de Google después de haber indexado documentos localmente) deja esos documentos inconsultables hasta resubirlos — limitación aceptada y documentada, sin migración/reindexado automático.

## Tests

```bash
pytest -v
```

37 tests contra el pipeline real (embeddings y ChromaDB reales, no mockeados) — mockear la propia mecánica de RAG habría dejado los tests con menos valor que la verificación manual. Los proveedores de LLM/embeddings de pago (Google, Anthropic) se prueban solo a nivel de selección/nombrado con keys falsas, sin llamada de red real — igual criterio para ambos, ninguno se ejercita de verdad en CI.

## Estructura

```
app/
├── main.py              # entrypoint, routers, CORS
├── core/                 # config
├── db/                    # base declarativa y sesión (SQLite, solo metadata)
├── models/                 # Document (SQLAlchemy)
├── schemas/                 # Pydantic schemas
├── api/routes/                # documents, search, ask, providers
├── services/                   # document_service (orquesta ingesta)
└── rag/                          # el corazón del proyecto
    ├── text_extraction.py         # pypdf
    ├── chunking.py                  # langchain-text-splitters
    ├── vector_store.py                # embeddings (local o Google) + ChromaDB, colección por proveedor
    └── llm_provider.py                  # Mock / Google / Anthropic, selección con override explícito
```

## Estado del proyecto / hitos

- [x] Ingesta de documentos (PDF/texto) con extracción real
- [x] Chunking real (no corte fijo ingenuo)
- [x] Embeddings locales o Google (según API key) + indexado en ChromaDB (métrica coseno, colección por proveedor)
- [x] Retrieval semántico con score de similitud
- [x] Generación con Gemini (por defecto) / Claude (alternativa explícita) / mock honesto (sin ninguna key)
- [x] Selección de proveedor por request (`POST /ask`) + endpoint de disponibilidad (`GET /providers`)
- [x] Tests (37, pipeline real) + CI

Pendiente:
- [ ] Deploy real — aplazado, necesita cuenta propia
- [ ] Frontend (repo independiente, mismo patrón que `expense-tracker-ui`)
- [ ] Alembic si el esquema de metadata crece (por ahora `create_all` es suficiente para una tabla)

## Bugs reales encontrados construyendo esto

1. **`pydantic-settings` 2.11** rompe el parseo de `List[str]` desde variables de entorno de una forma que `expense-api` (con 2.5.2) no sufría — mismo patrón de código, versión distinta, comportamiento distinto. Resuelto guardando el valor como `str` y exponiendo la lista vía `@property`.
2. **ChromaDB con métrica de distancia equivocada**: por defecto usa L2 al cuadrado, no coseno — el score de similitud no tenía sentido hasta configurar `hnsw:space: cosine` explícitamente. Detectado calculando el score de verdad, no solo comprobando que `search()` devolvía algo.
3. **Ruta `async def` con código bloqueante**: `sentence-transformers.encode()` corriendo directamente en una ruta `async def` congelaba el *event loop* entero del servidor — ninguna otra petición se atendía mientras tanto, ni siquiera `/health`. Solucionado usando `def` normal (FastAPI paraleliza rutas síncronas en un thread pool automáticamente).
4. **Colecciones de ChromaDB atadas a la dimensión del vector**: mezclar embeddings locales (384d, `sentence-transformers`) y de Google (768d) en la misma colección rompe las queries silenciosamente o lanza un error de dimensión — ChromaDB no permite dimensiones mixtas en una colección. Resuelto con una colección separada por proveedor+dimensión (`document_chunks_local_384d`, `document_chunks_google_768d`). Cambiar de proveedor de embeddings deja los documentos ya indexados con el proveedor anterior inconsultables hasta resubirlos — limitación aceptada y documentada, sin herramienta de migración/reindexado automático (fuera de alcance para un portfolio).
