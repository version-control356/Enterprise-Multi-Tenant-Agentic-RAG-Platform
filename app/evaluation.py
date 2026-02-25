"""Ragas evaluation utilities with deterministic latency reporting."""

from statistics import mean
from typing import Any, Sequence

from pydantic import BaseModel, Field


class RetrievalCase(BaseModel):
    """Expected retrieval result for one benchmark query."""

    query_id: str = Field(min_length=1)
    retrieved_ids: list[str]
    relevant_ids: set[str]
    latency_ms: float = Field(ge=0)


class RetrievalMetrics(BaseModel):
    """Aggregate Ragas retrieval quality and latency measurements."""

    cases: int
    context_precision: float
    context_recall: float
    mean_latency_ms: float
    p95_latency_ms: float


def validate_retrieval_cases(cases: Sequence[RetrievalCase]) -> None:
    """Reject empty benchmark input before invoking the Ragas evaluator."""
    if not cases:
        raise ValueError("At least one retrieval case is required.")


def evaluate_with_ragas(cases: Sequence[RetrievalCase], llm: Any) -> RetrievalMetrics:
    """Evaluate retrieved contexts with Ragas and append measured latency statistics."""
    validate_retrieval_cases(cases)
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import context_precision, context_recall
    except ImportError as error:
        raise RuntimeError("Ragas evaluation requires the ragas and datasets packages.") from error

    dataset = Dataset.from_dict(
        {
            "question": [case.query_id for case in cases],
            "contexts": [case.retrieved_ids for case in cases],
            "reference": [" ".join(sorted(case.relevant_ids)) for case in cases],
        }
    )
    result = evaluate(
        dataset,
        metrics=[context_precision, context_recall],
        llm=llm,
    )
    scores = result.to_pandas()
    latencies = [case.latency_ms for case in cases]
    sorted_latencies = sorted(latencies)
    p95_index = min(len(sorted_latencies) - 1, max(0, int(len(sorted_latencies) * 0.95) - 1))
    return RetrievalMetrics(
        cases=len(cases),
        context_precision=round(float(scores["context_precision"].mean()), 4),
        context_recall=round(float(scores["context_recall"].mean()), 4),
        mean_latency_ms=round(mean(latencies), 2),
        p95_latency_ms=round(sorted_latencies[p95_index], 2),
    )
