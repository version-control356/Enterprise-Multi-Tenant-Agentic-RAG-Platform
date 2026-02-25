"""Tests for Ragas evaluation input validation."""

import unittest

from app.evaluation import RetrievalCase, validate_retrieval_cases


class RetrievalEvaluationTests(unittest.TestCase):
    """Verify benchmark cases are validated before external evaluation."""

    def test_valid_cases_are_accepted(self) -> None:
        """Accept a labeled retrieval case without contacting an external evaluator."""
        validate_retrieval_cases(
            [
                RetrievalCase(
                    query_id="q1",
                    retrieved_ids=["wrong", "right"],
                    relevant_ids={"right"},
                    latency_ms=10,
                )
            ]
        )

    def test_empty_benchmark_is_rejected(self) -> None:
        """Reject empty evaluations so missing evidence cannot look like a passing benchmark."""
        with self.assertRaises(ValueError):
            validate_retrieval_cases([])
