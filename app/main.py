import argparse

from app.core.config import settings
from app.ingestion.chunker import chunk_text
from app.ingestion.cleaner import clean_text
from app.ingestion.loaders import load_document
from app.retrieval.retriever import retrieve


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Florence MVP - semantic document retrieval with Hugging Face"
    )
    parser.add_argument("--file", required=True, help="Path to the input document")
    parser.add_argument("--query", required=True, help="Literal string prompt")
    parser.add_argument("--top-k", type=int, default=5, help="How many chunks to return")

    args = parser.parse_args()

    raw_text = load_document(args.file)
    cleaned_text = clean_text(raw_text)
    chunks = chunk_text(
        cleaned_text,
        chunk_size=settings.chunk_size,
        overlap=settings.chunk_overlap,
    )

    response = retrieve(args.query, chunks, top_k=args.top_k)

    print(f"\nQuery: {response.query}")
    print(f"Chunks indexed: {len(chunks)}")
    print(f"Results found: {len(response.results)}\n")

    for i, item in enumerate(response.results, start=1):
        print("=" * 80)
        print(f"Result #{i}")
        print(f"Chunk ID: {item.metadata.get('chunk_id')}")
        print(f"Score: {item.metadata.get('score'):.4f}")
        print("-" * 80)
        print(item.document[:1500])
        print()


if __name__ == "__main__":
    main()