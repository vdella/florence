import logging
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.generation.qa import answer_from_contexts, get_qa_components
from app.generation.rewriter import get_rewriter_components, rewrite_answer
from app.ingestion.embedder import get_embedding_model
from app.ingestion.loaders import load_document
from app.retrieval.retriever import query_collection
from app.schemas.answers import AnswerResponse, CitationItem
from app.schemas.retrieval import SearchResponse
from app.services.ingestion_service import ingest_document_text
from app.storage.chroma_store import get_collection

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        get_embedding_model()
        logger.info("Embedding model loaded.")
    except Exception as exc:
        logger.exception("Embedding model preload failed: %r", exc)

    try:
        get_qa_components()
        logger.info("QA model loaded.")
    except Exception as exc:
        logger.exception("QA model preload failed: %r", exc)

    try:
        get_rewriter_components()
        logger.info("Rewriter model loaded.")
    except Exception as exc:
        logger.exception("Rewriter model preload failed: %r", exc)

    try:
        get_collection()
        logger.info("Chroma collection ready.")
    except Exception as exc:
        logger.exception("Chroma startup failed: %r", exc)

    yield


app = FastAPI(
    title="Florence API",
    version="0.5.0",
    description="Persistent RAG with Chroma, extractive QA and answer rewriting",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "Florence API is running",
        "docs": "/docs",
        "redoc": "/redoc",
    }


@app.get("/health")
def health() -> dict[str, int | str]:
    collection = get_collection()
    count = collection.count()
    return {
        "status": "ok",
        "indexed_chunks": count,
    }


@app.post("/ingest-file")
async def ingest_file(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".txt", ".pdf", ".docx"}:
        raise HTTPException(status_code=400, detail="supported file types: .txt, .pdf, .docx")

    temp_path = None

    try:
        content = await file.read()

        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            temp_path = tmp.name

        text, page_map = load_document(temp_path)
        source = file.filename or "uploaded_document"

        return ingest_document_text(
            text=text,
            source=source,
            page_map=page_map,
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=repr(exc)) from exc
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)


@app.post("/ingest-text")
def ingest_text(
        text: str = Form(...),
        source: str = Form(...),
) -> dict:
    try:
        return ingest_document_text(text=text, source=source, page_map=[{"page": 1, "text": text}])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=repr(exc)) from exc


@app.post("/query", response_model=SearchResponse)
def query(
        query: str = Form(...),
        top_k: int = Form(5),
) -> SearchResponse:
    if len(query.strip()) < 3:
        raise HTTPException(status_code=400, detail="query must have at least 3 characters")

    return query_collection(query=query, top_k=top_k)


@app.post("/ask", response_model=AnswerResponse)
def ask(
        query: str = Form(...),
        top_k: int = Form(5),
) -> AnswerResponse:
    if len(query.strip()) < 3:
        raise HTTPException(status_code=400, detail="query must have at least 3 characters")

    retrieval = query_collection(query=query, top_k=top_k)

    if not retrieval.results:
        return AnswerResponse(
            query=query,
            answer="Não encontrei informações relevantes na base.",
            extracted_answer="",
            answer_score=0.0,
            citations=[],
        )

    qa_contexts = [item.document for item in retrieval.results[:2]]
    qa_result = answer_from_contexts(question=query, contexts=qa_contexts)

    best_context = retrieval.results[0].document
    rewritten = rewrite_answer(
        question=query,
        extracted_answer=qa_result["answer"],
        context=best_context,
    )

    citations = [
        CitationItem(
            source=item.metadata.get("source"),
            chunk_id=item.metadata.get("chunk_id"),
            page=item.metadata.get("page"),
            score=(1.0 - item.distance) if item.distance is not None else None,
            excerpt=item.document[:300],
        )
        for item in retrieval.results
    ]

    return AnswerResponse(
        query=query,
        answer=rewritten,
        extracted_answer=qa_result["answer"],
        answer_score=float(qa_result["score"]),
        citations=citations,
    )