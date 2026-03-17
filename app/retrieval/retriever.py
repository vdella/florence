import chromadb

from app.core.config import settings
from app.ingestion.embedder import embed_texts


client = chromadb.PersistentClient(path=settings.chroma_dir)
collection = client.get_or_create_collection(name="patient_safety_docs")


def search_similar_chunks(query: str, top_k: int = 5) -> list[dict]:
    """
    Realiza busca semântica no banco vetorial e retorna os chunks mais relevantes.
    """
    query_embedding = embed_texts([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    response = []
    for doc, metadata, distance in zip(documents, metadatas, distances):
        response.append(
            {
                "document": doc,
                "metadata": metadata or {},
                "distance": float(distance) if distance is not None else None,
            }
        )

    return response