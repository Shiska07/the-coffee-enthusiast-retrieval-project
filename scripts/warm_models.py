from __future__ import annotations

from coffee_rag.config import settings


def warm_reranker() -> None:
    """Cross-encoder reranker. Downloads to the HF cache and runs one
    prediction to force the full weights load, not just the config fetch."""
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(settings.RERANKER_MODEL, device=settings.RERANKER_DEVICE)
    score = model.predict([("a coffee with bright acidity", "a juicy, citric Kenyan")])
    print(f"[reranker] {settings.RERANKER_MODEL} -> sample score {float(score[0]):.3f}")


def warm_hallucination() -> None:
    """MiniCheck factual-support classifier. Constructing it downloads + loads
    the weights; one score() call forces the forward pass."""
    from minicheck.minicheck import MiniCheck

    model = MiniCheck(model_name=settings.HALLUCINATION_MODEL)
    _, raw_prob, _, _ = model.score(docs=["The sky is blue."], claims=["The sky is blue."])
    print(f"[hallucination] {settings.HALLUCINATION_MODEL} -> sample score {float(raw_prob[0]):.3f}")


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
