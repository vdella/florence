from typing import Any


def chunk_text_with_metadata(
        text: str,
        source: str,
        chunk_size: int = 500,
        overlap: int = 80,
        page_map: list[dict] | None = None,
) -> list[dict[str, Any]]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size deve ser maior que overlap")

    chunks: list[dict[str, Any]] = []
    start = 0
    text_len = len(text)
    chunk_id = 0

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk_text = text[start:end].strip()

        if chunk_text:
            page = None
            if page_map:
                for item in page_map:
                    if chunk_text[:40].strip() and chunk_text[:40].strip() in item["text"]:
                        page = item.get("page")
                        break

            chunks.append(
                {
                    "id": f"{source}:{chunk_id}",
                    "document": chunk_text,
                    "metadata": {
                        "source": source,
                        "chunk_id": chunk_id,
                        "page": page,
                    },
                }
            )
            chunk_id += 1

        if end == text_len:
            break

        start = end - overlap

    return chunks