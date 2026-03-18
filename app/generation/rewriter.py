from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from app.core.config import settings

_rewriter_tokenizer = None
_rewriter_model = None


def get_rewriter_components():
    global _rewriter_tokenizer, _rewriter_model

    if _rewriter_tokenizer is None or _rewriter_model is None:
        model_name = settings.huggingface_rewriter_model
        _rewriter_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _rewriter_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        _rewriter_model.eval()

    return _rewriter_tokenizer, _rewriter_model


def rewrite_answer(question: str, extracted_answer: str, context: str) -> str:
    tokenizer, model = get_rewriter_components()

    prompt = f"""
Responda em português, de forma clara e curta, usando apenas o contexto abaixo.

Pergunta: {question}
Resposta extraída: {extracted_answer}
Contexto: {context}

Resposta final:
""".strip()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    output_ids = model.generate(
        **inputs,
        max_new_tokens=80,
        num_beams=4,
        early_stopping=True,
    )

    output = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    return output or extracted_answer