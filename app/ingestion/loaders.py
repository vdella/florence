from pathlib import Path

from docx import Document as DocxDocument
from pypdf import PdfReader


def load_pdf(path: Path) -> str:
    reader = PdfReader(path)
    text_parts: list[str] = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            text_parts.append(text)

    return "\n".join(text_parts)


def load_docx(path: Path) -> str:
    doc = DocxDocument(path)
    return "\n".join(p.text for p in doc.paragraphs)


def load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_document(path: str) -> str:
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