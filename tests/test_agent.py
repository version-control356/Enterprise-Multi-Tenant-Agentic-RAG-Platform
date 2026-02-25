"""Unit tests for the Corrective Agentic RAG graph topology."""

import unittest
from app.agents.graph import build_agent_graph, decide_to_generate, AgentState


class AgentGraphTests(unittest.TestCase):
    """Verify LangGraph compilation and conditional routing logic."""

    def test_graph_compiles_successfully(self) -> None:
        """Verify the state graph compiles without errors."""
        app = build_agent_graph(checkpointer=None)
        self.assertIsNotNone(app)

    def test_routing_decision_when_relevant(self) -> None:
        """Relevant context should route immediately to generation."""
        state: AgentState = {
            "messages": [],
            "tenant_id": "tenant-test",
            "user_role": "admin",
            "context": "Enterprise documentation context.",
            "query": "What is the policy?",
            "rewrite_count": 0,
            "is_relevant": True,
            "available_documents": ["policy.pdf"],
        }
        destination = decide_to_generate(state)
        self.assertEqual(destination, "generate")

    def test_routing_decision_when_disabled_goes_to_generate(self) -> None:
        """When query rewrite is disabled (default for low-latency), route directly to generate."""
        from app.config import settings
        settings.ENABLE_QUERY_REWRITE = False
        state: AgentState = {
            "messages": [],
            "tenant_id": "tenant-test",
            "user_role": "admin",
            "context": "",
            "query": "Vague question",
            "rewrite_count": 0,
            "is_relevant": False,
            "available_documents": ["notes.txt"],
        }
        destination = decide_to_generate(state)
        self.assertEqual(destination, "generate")

    def test_routing_decision_when_enabled_and_irrelevant(self) -> None:
        """When query rewrite is enabled and documents exist, route to rewrite_query."""
        from app.config import settings
        settings.ENABLE_QUERY_REWRITE = True
        try:
            state: AgentState = {
                "messages": [],
                "tenant_id": "tenant-test",
                "user_role": "admin",
                "context": "",
                "query": "Explain technical kubernetes cluster networking topology",
                "rewrite_count": 0,
                "is_relevant": False,
                "available_documents": ["k8s.pdf"],
            }
            destination = decide_to_generate(state)
            self.assertEqual(destination, "rewrite_query")
        finally:
            settings.ENABLE_QUERY_REWRITE = False

    def test_routing_decision_stops_after_max_rewrites(self) -> None:
        """After reaching maximum rewrites (>=1), route to generate to avoid infinite loops."""
        from app.config import settings
        settings.ENABLE_QUERY_REWRITE = True
        try:
            state: AgentState = {
                "messages": [],
                "tenant_id": "tenant-test",
                "user_role": "admin",
                "context": "",
                "query": "Vague question",
                "rewrite_count": 1,
                "is_relevant": False,
                "available_documents": ["k8s.pdf"],
            }
            destination = decide_to_generate(state)
            self.assertEqual(destination, "generate")
        finally:
            settings.ENABLE_QUERY_REWRITE = False

