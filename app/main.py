import os

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.routes import ask, documents, providers, search
from app.core.config import settings
from app.db.base import Base
from app.db.session import engine

os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(settings.chroma_persist_dir, exist_ok=True)

# No Alembic here (deliberate, unlike expense-api): a single small metadata
# table for a portfolio-scale personal tool doesn't earn the migration
# machinery — the point of this project is the RAG pipeline, not repeating
# migration setup already proven in expense-api.
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Personal RAG Assistant", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)
app.include_router(search.router)
app.include_router(ask.router)
app.include_router(providers.router)


@app.get("/health", tags=["health"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.exception_handler(StarletteHTTPException)
def http_exception_handler(request, exc: StarletteHTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


@app.exception_handler(RequestValidationError)
def validation_exception_handler(request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"error": "Validation failed", "details": exc.errors()})
