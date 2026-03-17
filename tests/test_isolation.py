"""Regression tests for tenant and conversation isolation."""

import unittest

from app.api.routes import build_backend_thread_id, build_backend_thread_scope, build_chat_cache_key
from app.config import Settings


class IsolationTests(unittest.TestCase):
    """Verify cache scopes cannot collide across security boundaries."""

    def test_cache_key_changes_for_user_and_thread(self) -> None:
        """Separate users, conversations, and cache versions must never share response cache entries."""
        base = build_chat_cache_key("tenant-a", "alice", "analyst", "thread-1", "question", cache_version=1)
        other_user = build_chat_cache_key(
            "tenant-a", "bob", "analyst", "thread-1", "question", cache_version=1
        )
        other_thread = build_chat_cache_key(
            "tenant-a", "alice", "analyst", "thread-2", "question", cache_version=1
        )
        other_version = build_chat_cache_key(
            "tenant-a", "alice", "analyst", "thread-1", "question", cache_version=2
        )

        self.assertNotEqual(base, other_user)
        self.assertNotEqual(base, other_thread)
        self.assertNotEqual(base, other_version)

    def test_thread_history_prefix_is_scoped_to_one_tenant_user(self) -> None:
        """Bulk history deletion must match only the caller's hashed thread namespace."""
        alice_prefix = build_backend_thread_scope("tenant-a", "alice")
        bob_prefix = build_backend_thread_scope("tenant-a", "bob")
        alice_thread = build_backend_thread_id("tenant-a", "alice", "thread-1")

        self.assertNotEqual(alice_prefix, bob_prefix)
        self.assertTrue(alice_thread.startswith(alice_prefix[:-1]))

    def test_jwt_algorithm_is_restricted(self) -> None:
        """Reject unsafe or unsupported JWT algorithm configuration."""
        with self.assertRaises(ValueError):
            Settings(SECRET_KEY="x" * 32, ALGORITHM="none")

    def test_telemetry_tracker_clear_traces(self) -> None:
        """Verify tenant-scoped and global trace purging."""
        from app.core.tracing import TelemetryTracker
        tracker = TelemetryTracker()
        tracker.finalize_trace(
            trace_id="t1",
            tenant_id="tenant_a",
            user_id="user_1",
            role="admin",
            query="test query a",
            start_time=100.0,
        )
        tracker.finalize_trace(
            trace_id="t2",
            tenant_id="tenant_b",
            user_id="user_2",
            role="admin",
            query="test query b",
            start_time=100.0,
        )
        self.assertEqual(len(tracker.get_traces_for_tenant("tenant_a")), 1)
        self.assertEqual(len(tracker.get_traces_for_tenant("tenant_b")), 1)

        purged = tracker.clear_traces("tenant_a")
        self.assertEqual(purged, 1)
        self.assertEqual(len(tracker.get_traces_for_tenant("tenant_a")), 0)
        self.assertEqual(len(tracker.get_traces_for_tenant("tenant_b")), 1)

        tracker.clear_traces()
        self.assertEqual(len(tracker.get_traces_for_tenant("tenant_b")), 0)

