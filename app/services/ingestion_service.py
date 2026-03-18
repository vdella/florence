from app.core.config import settings
from app.ingestion.chunker import chunk_text_with_metadata
from app.ingestion.cleaner import clean_text
from app.ingestion.embedder import embed_texts
from app.storage.chroma_store import get_collection


def ingest_document_text(text: str, source: str, page_map: list[dict] | None = None) -> dict:
    cleaned = clean_text(text)
    if not cleaned:
        raise ValueError("document is empty after cleaning")

    chunks = chunk_text_with_metadata(
        text=cleaned,
        source=source,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
        page_map=page_map,
    )

    if not chunks:
        raise ValueError("document produced no usable chunks")

    collection = get_collection()

    ids = [chunk["id"] for chunk in chunks]
    documents = [chunk["document"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]
    embeddings = embed_texts(documents)

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    return {
        "source": source,
        "chunks_ingested": len(chunks),
    }