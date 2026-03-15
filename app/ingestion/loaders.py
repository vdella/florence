from pathlib import Path
from pypdf import PdfReader
from docx import Document as DocxDocument


def load_pdf(path: Path) -> str:
    reader = PdfReader(path)

    text_parts = []

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
    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        return load_pdf(file_path)

    if suffix == ".docx":
        return load_docx(file_path)

    if suffix == ".txt":
        return load_txt(file_path)

    raise ValueError(f"Formato não suportado: {suffix}")


if __name__ == "__main__":
    resources = "/home/della/Vault/florence/resources/"
    tcc = Path(resources + "tcc.pdf")
    workplan = Path(resources + "workplan.pdf")

    print(load_pdf(tcc))
    print(load_pdf(workplan))