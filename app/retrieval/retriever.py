from app.ingestion.embedder import embed_query
from app.schemas.retrieval import SearchResponse, SearchResultItem
from app.storage.chroma_store import get_collection


def query_collection(query: str, top_k: int = 5) -> SearchResponse:
    collection = get_collection()
    query_embedding = embed_query(query)

    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    items = []
    for document, metadata, distance in zip(documents, metadatas, distances):
        items.append(
            SearchResultItem(
                document=document,
                metadata=metadata or {},
                distance=float(distance) if distance is not None else None,
            )
        )

    return SearchResponse(query=query, results=items)