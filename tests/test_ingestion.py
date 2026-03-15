from pathlib import Path

import pytest

from app.ingestion.loaders import load_txt, load_document
from app.ingestion.cleaner import clean_text
from app.ingestion.chunker import chunk_text


def test_load_txt_reads_file_content(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.txt"
    file_path.write_text("Protocolo de segurança do paciente.", encoding="utf-8")

    content = load_txt(file_path)

    assert content == "Protocolo de segurança do paciente."


def test_load_document_supports_txt(tmp_path: Path) -> None:
    file_path = tmp_path / "documento.txt"
    file_path.write_text("Plano de ação hospitalar.", encoding="utf-8")

    content = load_document(str(file_path))

    assert content == "Plano de ação hospitalar."


def test_load_document_raises_for_unsupported_extension(tmp_path: Path) -> None:
    file_path = tmp_path / "arquivo.csv"
    file_path.write_text("coluna1,coluna2", encoding="utf-8")

    with pytest.raises(ValueError, match="Formato não suportado"):
        load_document(str(file_path))


def test_clean_text_removes_extra_spaces_and_lines() -> None:
    raw_text = "Linha 1\r\n\r\n\r\nLinha 2    com   espaços\n\n\nLinha 3"

    cleaned = clean_text(raw_text)

    assert cleaned == "Linha 1\n\nLinha 2 com espaços\n\nLinha 3"


def test_clean_text_removes_hyphen_line_break() -> None:
    raw_text = "segu-\nrança do paciente"

    cleaned = clean_text(raw_text)

    assert cleaned == "segurança do paciente"


def test_chunk_text_splits_long_text() -> None:
    text = "A" * 2000

    chunks = chunk_text(text, chunk_size=800, overlap=100)

    assert len(chunks) >= 3
    assert all(isinstance(chunk, str) for chunk in chunks)
    assert all(len(chunk) <= 800 for chunk in chunks)


def test_chunk_text_keeps_small_text_in_single_chunk() -> None:
    text = "Texto curto para teste."

    chunks = chunk_text(text, chunk_size=800, overlap=100)

    assert chunks == ["Texto curto para teste."]


def test_chunk_text_raises_when_overlap_is_invalid() -> None:
    with pytest.raises(ValueError, match="chunk_size deve ser maior que overlap"):
        chunk_text("texto qualquer", chunk_size=100, overlap=100)