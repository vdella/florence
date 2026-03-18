import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from app.core.config import settings

_tokenizer = None
_model = None


def get_qa_components():
    global _tokenizer, _model

    if _tokenizer is None or _model is None:
        model_name = settings.huggingface_qa_model

        try:
            _tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        except Exception:
            _tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

        _model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        _model.eval()

    return _tokenizer, _model


def extract_answer(question: str, context: str) -> dict:
    tokenizer, model = get_qa_components()

    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    start_logits = outputs.start_logits[0]
    end_logits = outputs.end_logits[0]

    start_idx = int(torch.argmax(start_logits))
    end_idx = int(torch.argmax(end_logits))

    if end_idx < start_idx:
        end_idx = start_idx

    answer_ids = inputs["input_ids"][0][start_idx : end_idx + 1]
    answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

    start_score = float(torch.softmax(start_logits, dim=0)[start_idx])
    end_score = float(torch.softmax(end_logits, dim=0)[end_idx])
    score = (start_score + end_score) / 2.0

    return {
        "answer": answer,
        "score": score,
        "start": start_idx,
        "end": end_idx,
    }


def answer_from_contexts(question: str, contexts: list[str], min_score: float | None = None) -> dict:
    if min_score is None:
        min_score = settings.qa_min_score

    best_result = None
    best_context_index = None

    for idx, context in enumerate(contexts):
        if not context.strip():
            continue

        result = extract_answer(question=question, context=context)
        score = float(result.get("score", 0.0))

        if best_result is None or score > float(best_result.get("score", 0.0)):
            best_result = result
            best_context_index = idx

    if (
            best_result is None
            or float(best_result.get("score", 0.0)) < min_score
            or not str(best_result.get("answer", "")).strip()
    ):
        return {
            "answer": "Não encontrei uma resposta confiável no documento.",
            "score": 0.0,
            "context_index": None,
        }

    return {
        "answer": str(best_result["answer"]).strip(),
        "score": float(best_result["score"]),
        "context_index": best_context_index,
    }