from pydantic import BaseModel, Field
from typing import Any


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3)
    top_k: int = Field(default=5, ge=1, le=20)


class SearchResultItem(BaseModel):
    document: str
    metadata: dict[str, Any]
    distance: float | None = None


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResultItem]