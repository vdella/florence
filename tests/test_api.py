from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"


def test_query_text():
    response = client.post(
        "/query-text",
        data={
            "text": (
                "Sepse é uma condição clínica grave. "
                "Os sinais incluem febre, confusão mental e hipotensão."
            ),
            "query": "quais são os sinais de sepse?",
            "top_k": 2,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "quais são os sinais de sepse?"
    assert len(payload["results"]) >= 1


def test_ask_text():
    response = client.post(
        "/ask-text",
        data={
            "text": (
                "Sepse é uma condição clínica grave. "
                "Os sinais incluem febre, confusão mental e hipotensão."
            ),
            "query": "quais são os sinais de sepse?",
            "top_k": 2,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "quais são os sinais de sepse?"
    assert payload["answer"]
    assert payload["answer_score"] >= 0.0