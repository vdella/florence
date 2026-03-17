from app.retrieval.retriever import retrieve


def test_retrieve_returns_best_match():
    chunks = [
        "Diabetes mellitus pode causar hiperglicemia persistente.",
        "Sepse pode apresentar febre, confusão mental e hipotensão.",
        "Fraturas ósseas exigem avaliação radiológica.",
    ]

    response = retrieve("quais são os sinais de sepse?", chunks, top_k=2)

    print(response)

    assert len(response.results) >= 1
    assert "Sepse" in response.results[0].document