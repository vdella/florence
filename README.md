# florence

MVP de leitura de documentos com busca semântica usando Hugging Face.

## O que este projeto faz

- carrega arquivos `.txt`, `.pdf` e `.docx`
- limpa o texto
- divide em chunks
- gera embeddings com Hugging Face (`sentence-transformers`)
- recebe uma consulta literal em string
- retorna os chunks semanticamente mais próximos

## Stack inicial

- Python 3.11+
- `pypdf`
- `python-docx`
- `pydantic`
- `pydantic-settings`
- `sentence-transformers`

## Instalação

```bash
uv sync