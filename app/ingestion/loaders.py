from pathlib import Path

from docx import Document as DocxDocument
from pypdf import PdfReader


def load_pdf(path: Path) -> tuple[str, list[dict]]:
    reader = PdfReader(path)
    text_parts: list[str] = []
    page_map: list[dict] = []

    for page_idx, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            page_map.append(
                {
                    "page": page_idx + 1,
                    "text": text,
                }
            )
            text_parts.append(text)

    return "\n".join(text_parts), page_map


def load_docx(path: Path) -> tuple[str, list[dict]]:
    doc = DocxDocument(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    text = "\n".join(paragraphs)
    return text, [{"page": 1, "text": text}] if text.strip() else []


def load_txt(path: Path) -> tuple[str, list[dict]]:
    text = path.read_text(encoding="utf-8")
    return text, [{"page": 1, "text": text}] if text.strip() else []


def load_document(path: str) -> tuple[str, list[dict]]:
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return load_pdf(file_path)
    if suffix == ".docx":
        return load_docx(file_path)
    if suffix == ".txt":
        return load_txt(file_path)

    raise ValueError(f"Formato não suportado: {suffix}")