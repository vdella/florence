from pydantic import BaseModel


class CitationItem(BaseModel):
    source: str | None = None
    chunk_id: int | None = None
    page: int | None = None
    score: float | None = None
    excerpt: str


class AnswerResponse(BaseModel):
    query: str
    answer: str
    extracted_answer: str
    answer_score: float
    citations: list[CitationItem]