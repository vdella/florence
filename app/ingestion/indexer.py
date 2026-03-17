import uuid
from pathlib import Path
import chromadb

from app.core.config import settings
from app.ingestion.loaders import load_document
from app.ingestion.cleaner import clean_text
from app.ingestion.chunker import chunk_text
from app.ingestion.embedder import embed_texts


client = chromadb.PersistentClient(path=settings.chroma_dir)
collection = client.get_or_create_collection(name="patient_safety_docs")


def index_file(file_path: str) -> int:
    raw_text = load_document(file_path)
    cleaned = clean_text(raw_text)
    chunks = chunk_text(
        cleaned,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )

    embeddings = embed_texts(chunks)

    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [{"source": Path(file_path).name, "chunk_index": i} for i, _ in enumerate(chunks)]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return len(chunks)