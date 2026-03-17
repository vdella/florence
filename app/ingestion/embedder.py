from sentence_transformers import SentenceTransformer

from app.core.config import settings

_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    global _model

    if _model is None:
        _model = SentenceTransformer(settings.huggingface_embedding_model)

    return _model


def embed_texts(texts: list[str]):
    model = get_embedding_model()
    return model.encode(
        texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def embed_query(query: str):
    model = get_embedding_model()
    embeddings = model.encode(
        [query],
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings[0]