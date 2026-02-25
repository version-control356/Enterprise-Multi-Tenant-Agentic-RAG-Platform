"""Run the reproducible Ragas retrieval evaluation fixture."""

import json

from app.config import settings
from app.evaluation import RetrievalCase, evaluate_with_ragas


def build_evaluator() -> object:
    """Create the LangChain LLM adapter used by Ragas."""
    if not settings.GROQ_API_KEY.strip():
        raise RuntimeError("GROQ_API_KEY is required to run the Ragas evaluation.")
    from langchain_groq import ChatGroq

    return ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_MODEL,
        temperature=0,
    )


def main() -> None:
    """Evaluate the bundled smoke-test retrieval cases with Ragas."""
    cases = [
        RetrievalCase(
            query_id="q1",
            retrieved_ids=["policy-1", "policy-2", "faq-1"],
            relevant_ids={"policy-1"},
            latency_ms=18.0,
        ),
        RetrievalCase(
            query_id="q2",
            retrieved_ids=["sku-2", "sku-1", "sku-3"],
            relevant_ids={"sku-1"},
            latency_ms=22.0,
        ),
        RetrievalCase(
            query_id="q3",
            retrieved_ids=["security-1", "security-2"],
            relevant_ids={"security-1"},
            latency_ms=20.0,
        ),
    ]
    print(json.dumps(evaluate_with_ragas(cases, build_evaluator()).model_dump(), indent=2))


if __name__ == "__main__":
    main()
