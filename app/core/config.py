from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    huggingface_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    huggingface_qa_model: str = "deepset/xlm-roberta-base-squad2-distilled"
    huggingface_rewriter_model: str = "google/flan-t5-small"

    chroma_path: str = "data/chroma"
    chroma_collection_name: str = "documents"

    chunk_size: int = 500
    chunk_overlap: int = 80

    retrieval_top_k_default: int = 5
    qa_top_k_contexts: int = 2
    qa_min_score: float = 0.05

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()