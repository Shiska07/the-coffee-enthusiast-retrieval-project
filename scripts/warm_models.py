from __future__ import annotations

from coffee_rag.config import settings


def warm_reranker() -> None:
    """Stage-1 cross-encoder reranker. Downloads to the HF cache and runs one
    prediction to force the full weights load, not just the config fetch."""
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(settings.RERANKER_MODEL, device=settings.RERANKER_DEVICE)
    score = model.predict([("a coffee with bright acidity", "a juicy, citric Kenyan")])
    print(f"[reranker] {settings.RERANKER_MODEL} -> sample score {float(score[0]):.3f}")


def warm_hallucination() -> None:
    """Vectara HHEM factual-consistency classifier. ``trust_remote_code`` is
    required: the repo ships a custom model class with its own ``predict``."""
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        settings.HALLUCINATION_MODEL, trust_remote_code=True
    )
    score = model.predict([("The sky is blue.", "The sky is blue.")])
    print(f"[hallucination] {settings.HALLUCINATION_MODEL} -> sample score {float(score[0]):.3f}")


WARMERS = {
    "reranker": warm_reranker,
    "hallucination": warm_hallucination,
}


def main() -> int:
    failed: list[str] = []
    for name,warm in WARMERS.items():
        print(f"warming {name} ...")
        try:
            warm()
        except Exception as exc:
            print(f"  {name} FAILED: {exc!r}")
            failed.append(name)

    print()
    if failed:
        print(f"done with failures: {', '.join(failed)}")
        return 1
    print("all models warm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
