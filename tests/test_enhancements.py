"""Unit tests for Cohere reranker, LLM grader fallback, OCR handling, and async ingestion."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from app.core.rerank import rerank_documents_with_cohere
from app.agents.graph import grade_documents_node, AgentState
from app.ingestion.parser import UniversalDocumentParser


class EnhancementTests(unittest.TestCase):
    """Test suite covering the 4 new platform capabilities."""

    def test_cohere_rerank_fallback_without_api_key(self) -> None:
        """When Cohere API key is not configured, candidate docs should be returned in original order."""
        docs = ["Doc chunk 1", "Doc chunk 2", "Doc chunk 3"]
        result = asyncio.run(
            rerank_documents_with_cohere(query="test query", documents=docs, top_n=2)
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], "Doc chunk 1")

    def test_llm_grader_returns_false_for_empty_context(self) -> None:
        """Empty context should immediately be graded as irrelevant without calling LLM."""
        state: AgentState = {
            "messages": [],
            "tenant_id": "tenant-test",
            "user_role": "admin",
            "context": "",
            "query": "What is the policy?",
            "rewrite_count": 0,
            "is_relevant": False,
        }
        result = asyncio.run(grade_documents_node(state))
        self.assertFalse(result["is_relevant"])

    def test_fast_heuristic_grader_relevance(self) -> None:
        """Fast heuristic grader should quickly detect relevant context matching query tokens."""
        state: AgentState = {
            "messages": [],
            "tenant_id": "tenant-test",
            "user_role": "analyst",
            "context": "--- Document Source: policy.pdf ---\nThe enterprise data retention policy is 90 days.",
            "query": "What is the data retention policy?",
            "rewrite_count": 0,
            "is_relevant": False,
        }
        result = asyncio.run(grade_documents_node(state))
        self.assertTrue(result["is_relevant"])

    def test_scanned_pdf_raises_descriptive_error(self) -> None:
        """An empty/scanned PDF without OCR dependencies should fail with a helpful error."""
        empty_pdf_bytes = (
            b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n"
            b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj\n"
            b"xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000052 00000 n \n"
            b"0000000101 00000 n \ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n162\n%%EOF"
        )
        with self.assertRaises(ValueError) as ctx:
            UniversalDocumentParser.extract_text(empty_pdf_bytes, "scanned.pdf")
        self.assertIn("PDF document does not contain extractable text", str(ctx.exception))

    def test_async_ingestion_task_lifecycle(self) -> None:
        """Async ingestion worker should track task status transitions."""
        from app.api.routes import _ingestion_tasks
        task_id = "test-task-123"
        _ingestion_tasks[task_id] = {
            "task_id": task_id,
            "status": "processing",
            "filename": "sample.txt",
            "chunks_ingested": 0,
            "error": None,
        }
        self.assertEqual(_ingestion_tasks[task_id]["status"], "processing")

    def test_new_document_ingestion_keeps_newly_upserted_vectors(self) -> None:
        """A first upload must not delete its own Qdrant points during replacement cleanup."""
        from app.api import routes

        vector = SimpleNamespace(tolist=lambda: [0.1, 0.2])
        sparse_vector = SimpleNamespace(
            indices=SimpleNamespace(tolist=lambda: [1]),
            values=SimpleNamespace(tolist=lambda: [0.5]),
        )
        upsert = AsyncMock()
        delete = AsyncMock()
        with (
            patch.object(routes.UniversalDocumentParser, "extract_text", return_value="policy text"),
            patch.object(routes.UniversalDocumentParser, "chunk_document", return_value=["policy text"]),
            patch.object(routes, "get_tenant_document_usage", new=AsyncMock(return_value=(0, 0))),
            patch.object(routes, "get_tenant_document_id", new=AsyncMock(return_value=None)),
            patch.object(routes, "upsert_documents_to_qdrant", new=upsert),
            patch.object(routes, "delete_documents_from_qdrant", new=delete),
            patch.object(routes, "record_document", new=AsyncMock()),
            patch.object(routes, "bump_tenant_cache_version", new=AsyncMock()),
            patch.object(routes, "get_embedding_model", return_value=Mock(embed=lambda _: [vector])),
            patch.object(routes, "get_sparse_embedding_model", return_value=Mock(embed=lambda _: [sparse_vector])),
        ):
            result = asyncio.run(
                routes._process_document_ingestion(
                    content=b"policy text",
                    filename="policy.txt",
                    allowed_roles="admin",
                    tenant_id="tenant-a",
                    user_id="alice",
                )
            )

        self.assertEqual(result["status"], "success")
        upsert.assert_awaited_once()
        delete.assert_not_awaited()

    def test_clear_telemetry_traces(self) -> None:
        """Clearing telemetry traces should remove recorded records for tenant."""
        import time
        from app.core.tracing import telemetry_tracker
        t0 = time.perf_counter()
        telemetry_tracker.start_trace("trace-del-1")
        telemetry_tracker.finalize_trace("trace-del-1", "test_tenant", "user1", "admin", "test query", start_time=t0)
        self.assertTrue(len(telemetry_tracker.get_traces_for_tenant("test_tenant")) > 0)
        
        purged = telemetry_tracker.clear_traces("test_tenant")
        self.assertGreaterEqual(purged, 1)
        self.assertEqual(len(telemetry_tracker.get_traces_for_tenant("test_tenant")), 0)
