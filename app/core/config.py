from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    huggingface_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    huggingface_qa_model: str = "deepset/xlm-roberta-base-squad2-distilled"

    raw_data_dir: str = "data/raw"
    chunk_size: int = 500
    chunk_overlap: int = 80

    qa_top_k_contexts: int = 1
    qa_min_score: float = 0.05

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()