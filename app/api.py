import hashlib
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.core.config import settings
from app.generation.qa import answer_from_contexts, get_qa_components
from app.ingestion.chunker import chunk_text
from app.ingestion.cleaner import clean_text
from app.ingestion.embedder import embed_texts, get_embedding_model
from app.ingestion.loaders import load_document
from app.retrieval.retriever import retrieve, retrieve_from_embeddings
from app.retrieval.store import CachedDocument, DOCUMENT_CACHE
from app.schemas.retrieval import SearchResponse

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        emb_model = get_embedding_model()
        emb_model.encode(["warmup"], show_progress_bar=False)
        logger.info("Embedding model loaded successfully.")
    except Exception as exc:
        logger.exception("Embedding model preload failed: %r", exc)

    try:
        get_qa_components()
        answer_from_contexts(
            question="What is sepsis?",
            contexts=["Sepsis is a serious condition caused by infection."],
        )
        logger.info("QA model loaded successfully.")
    except Exception as exc:
        logger.exception("QA model preload failed: %r", exc)

    yield


app = FastAPI(
    title="Florence API",
    version="0.4.0",
    description="Document reading MVP with Hugging Face retrieval + extractive QA + file cache",
    lifespan=lifespan,
)


class AnswerContextItem(BaseModel):
    chunk_id: int | None = None
    score: float | None = None
    document: str


class AnswerResponse(BaseModel):
    query: str
    answer: str
    answer_score: float
    context: list[AnswerContextItem]


def _hash_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _load_and_prepare_document_from_temp_path(temp_path: str) -> tuple[str, list[str]]:
    raw_text = load_document(temp_path)
    cleaned_text = clean_text(raw_text)

    if not cleaned_text:
        raise HTTPException(status_code=400, detail="document is empty after extraction/cleaning")

    chunks = chunk_text(
        cleaned_text,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )

    if not chunks:
        raise HTTPException(status_code=400, detail="document produced no usable chunks")

    return cleaned_text, chunks


def _get_or_create_cached_document(file_bytes: bytes, suffix: str) -> CachedDocument:
    file_hash = _hash_bytes(file_bytes)

    if file_hash in DOCUMENT_CACHE:
        return DOCUMENT_CACHE[file_hash]

    temp_path: str | None = None
    try:
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            temp_path = tmp.name

        _, chunks = _load_and_prepare_document_from_temp_path(temp_path)
        embeddings = embed_texts(chunks)

        cached = CachedDocument(
            chunks=chunks,
            embeddings=embeddings,
        )
        DOCUMENT_CACHE[file_hash] = cached
        return cached
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)


def _build_answer_response_from_chunks_and_embeddings(
        query: str,
        chunks: list[str],
        embeddings,
        top_k: int,
) -> AnswerResponse:
    retrieval = retrieve_from_embeddings(
        query=query,
        chunks=chunks,
        chunk_embeddings=embeddings,
        top_k=top_k,
    )

    if not retrieval.results:
        return AnswerResponse(
            query=query,
            answer="Não encontrei informações relevantes no documento.",
            answer_score=0.0,
            context=[],
        )

    qa_contexts = [item.document for item in retrieval.results[: settings.qa_top_k_contexts]]
    qa_result = answer_from_contexts(question=query, contexts=qa_contexts)

    return AnswerResponse(
        query=query,
        answer=qa_result["answer"],
        answer_score=float(qa_result["score"]),
        context=[
            AnswerContextItem(
                chunk_id=item.metadata.get("chunk_id"),
                score=item.metadata.get("score"),
                document=item.document,
            )
            for item in retrieval.results
        ],
    )


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "Florence API is running",
        "docs": "/docs",
        "redoc": "/redoc",
    }


@app.get("/health")
def health() -> dict[str, str | int]:
    return {
        "status": "ok",
        "cached_documents": len(DOCUMENT_CACHE),
    }


@app.post("/query-text", response_model=SearchResponse)
def query_text(
        text: str = Form(...),
        query: str = Form(...),
        top_k: int = Form(5),
) -> SearchResponse:
    if len(query.strip()) < 3:
        raise HTTPException(status_code=400, detail="query must have at least 3 characters")

    cleaned_text = clean_text(text)
    if not cleaned_text:
        raise HTTPException(status_code=400, detail="provided text is empty after cleaning")

    chunks = chunk_text(
        cleaned_text,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )

    return retrieve(query=query, chunks=chunks, top_k=top_k)


@app.post("/ask-text", response_model=AnswerResponse)
def ask_text(
        text: str = Form(...),
        query: str = Form(...),
        top_k: int = Form(5),
) -> AnswerResponse:
    if len(query.strip()) < 3:
        raise HTTPException(status_code=400, detail="query must have at least 3 characters")

    cleaned_text = clean_text(text)
    if not cleaned_text:
        raise HTTPException(status_code=400, detail="provided text is empty after cleaning")

    try:
        chunks = chunk_text(
            cleaned_text,
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )
        embeddings = embed_texts(chunks)

        return _build_answer_response_from_chunks_and_embeddings(
            query=query,
            chunks=chunks,
            embeddings=embeddings,
            top_k=top_k,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=repr(exc)) from exc


@app.post("/query-file", response_model=SearchResponse)
async def query_file(
        file: UploadFile = File(...),
        query: str = Form(...),
        top_k: int = Form(5),
) -> SearchResponse:
    if len(query.strip()) < 3:
        raise HTTPException(status_code=400, detail="query must have at least 3 characters")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".txt", ".pdf", ".docx"}:
        raise HTTPException(status_code=400, detail="supported file types: .txt, .pdf, .docx")

    try:
        file_bytes = await file.read()
        cached = _get_or_create_cached_document(file_bytes=file_bytes, suffix=suffix)

        return retrieve_from_embeddings(
            query=query,
            chunks=cached.chunks,
            chunk_embeddings=cached.embeddings,
            top_k=top_k,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=repr(exc)) from exc


@app.post("/ask-file", response_model=AnswerResponse)
async def ask_file(
        file: UploadFile = File(...),
        query: str = Form(...),
        top_k: int = Form(5),
) -> AnswerResponse:
    if len(query.strip()) < 3:
        raise HTTPException(status_code=400, detail="query must have at least 3 characters")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".txt", ".pdf", ".docx"}:
        raise HTTPException(status_code=400, detail="supported file types: .txt, .pdf, .docx")

    try:
        file_bytes = await file.read()
        cached = _get_or_create_cached_document(file_bytes=file_bytes, suffix=suffix)

        return _build_answer_response_from_chunks_and_embeddings(
            query=query,
            chunks=cached.chunks,
            embeddings=cached.embeddings,
            top_k=top_k,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=repr(exc)) from exc