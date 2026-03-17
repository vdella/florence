from sentence_transformers import util

from app.ingestion.embedder import embed_query, embed_texts
from app.schemas.retrieval import SearchResponse, SearchResultItem


def retrieve(query: str, chunks: list[str], top_k: int = 5) -> SearchResponse:
    if not chunks:
        return SearchResponse(query=query, results=[])

    chunk_embeddings = embed_texts(chunks)
    return retrieve_from_embeddings(
        query=query,
        chunks=chunks,
        chunk_embeddings=chunk_embeddings,
        top_k=top_k,
    )


def retrieve_from_embeddings(
        query: str,
        chunks: list[str],
        chunk_embeddings,
        top_k: int = 5,
) -> SearchResponse:
    if not chunks:
        return SearchResponse(query=query, results=[])

    query_embedding = embed_query(query)
    scores = util.cos_sim(query_embedding, chunk_embeddings)[0]

    k = min(top_k, len(chunks))
    top_indices = scores.argsort(descending=True)[:k].tolist()

    results: list[SearchResultItem] = []
    for idx in top_indices:
        score = float(scores[idx])

        results.append(
            SearchResultItem(
                document=chunks[idx],
                metadata={
                    "chunk_id": idx,
                    "score": score,
                },
                distance=1.0 - score,
            )
        )

    return SearchResponse(query=query, results=results)