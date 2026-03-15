from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str
    embedding_model: str = "text-embedding-3-small"
    chroma_dir: str = "data/chroma"
    raw_data_dir: str = "data/raw"
    chunk_size: int = 800
    chunk_overlap: int = 120

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()
